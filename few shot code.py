import sys, json, time, csv, re
from typing import List, Dict, Any
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator
from openai import AzureOpenAI

AZURE_OPENAI_ENDPOINT    = "https://staging-openai.azure-api.net/openai-gw-proxy-dev/"
AZURE_OPENAI_API_VERSION = "2025-02-01-preview"
AZURE_OPENAI_API_KEY     = "b5844b8ef1d3437dbff2dfd109a4034a"
AZURE_OPENAI_DEPLOYMENT  = "gpt-5-2025-08-07"

BATCH_SIZE     = 20     # rows per request
MAX_RETRIES    = 3
MAX_TEXT_CHARS = 2000   # defensive truncation
LABELS = ["joy", "other", "sadness", "anger", "disgust", "surprise", "fear"]

# ───────────────────────── Output schema (for validation) ─────────────────────────── #
class ItemOut(BaseModel):
    id: str
    labels: List[str]

    @field_validator("labels")
    @classmethod
    def check_labels(cls, v):
        allowed, unique = set(LABELS), []
        for lab in v:
            if lab not in allowed:
                raise ValueError(f"label '{lab}' not in {allowed}")
            if lab not in unique:
                unique.append(lab)
        return unique

class BatchOut(BaseModel):
    items: List[ItemOut]

# ───────────────────────────── Prompt templates ───────────────────────────── #

# Version 1: WITH EMOJIS
SYSTEM_MSG_WITH_EMOJIS = f"""You are a careful multi-label emotion classifier.

Classify each input text into zero or more of these labels:
{LABELS}

Rules:
- A text may receive 1 or several labels.
- Use "other" label for text that expresses emotions not covered by the specific categories (joy, sadness, anger, disgust, surprise, fear).
- Use "other" for neutral or ambiguous emotional content.
- Treat each item independently.
- Preserve every input id exactly.

Here are some examples:

Example 1:
Input: {{"id": "319", "text": "I remember when Rooney wanted to leave United and the fans threaten to kill the man and bare junk... wanna regretting that now? 😂😂"}}
Output: {{"id": "319", "labels": ["anger", "disgust"]}}

Example 2:
Input: {{"id": "1557", "text": "Epic battle HASHTAG breathtaking, Arya is awesome! 👍"}}
Output: {{"id": "1557", "labels": ["joy"]}}

Example 3:
Input: {{"id": "2005", "text": "HASHTAG Glory ,then ashesMen's dreams in a spark transformedBurning hearts unite 🌹🙏"}}
Output: {{"id": "2005", "labels": ["other"]}}

Return your response as a JSON object with this exact structure:
{{
  "items": [
    {{"id": "item_id", "labels": ["label1", "label2"]}},
    {{"id": "item_id", "labels": ["other"]}}
  ]
}}

IMPORTANT: Return ONLY the JSON object, no additional text or formatting."""

# Version 2: WITHOUT EMOJIS
SYSTEM_MSG_WITHOUT_EMOJIS = f"""You are a careful multi-label emotion classifier.

Classify each input text into zero or more of these labels:
{LABELS}

Rules:
- A text may receive 1 or several labels.
- Use "other" label for text that expresses emotions not covered by the specific categories (joy, sadness, anger, disgust, surprise, fear).
- Use "other" for neutral or ambiguous emotional content.
- Treat each item independently.
- Preserve every input id exactly.

Here are some examples:

Example 1:
Input: {{"id": "319", "text": "I remember when Rooney wanted to leave United and the fans threaten to kill the man and bare junk... wanna regretting that now?"}}
Output: {{"id": "319", "labels": ["anger", "disgust"]}}

Example 2:
Input: {{"id": "1557", "text": "Epic battle HASHTAG breathtaking, Arya is awesome!"}}
Output: {{"id": "1557", "labels": ["joy"]}}

Example 3:
Input: {{"id": "2005", "text": "HASHTAG Glory ,then ashesMen's dreams in a spark transformedBurning hearts unite"}}
Output: {{"id": "2005", "labels": ["other"]}}

Return your response as a JSON object with this exact structure:
{{
  "items": [
    {{"id": "item_id", "labels": ["label1", "label2"]}},
    {{"id": "item_id", "labels": ["other"]}}
  ]
}}

IMPORTANT: Return ONLY the JSON object, no additional text or formatting."""

USER_TEMPLATE = """Classify the emotions in each text:

{items_json}

Remember to return ONLY a JSON object with the structure shown in the system message."""

# ────────────────────────── Azure helper functions ────────────────────────── #
def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

# ───────────────────────────── Utility helpers ────────────────────────────── #
def contains_emojis(df):
    """Check if the dataset contains emojis"""
    import re
    emoji_pattern = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              u"\U00002702-\U000027B0"
                              u"\U000024C2-\U0001F251"
                              "]+", flags=re.UNICODE)
    
    sample_texts = df['text'].head(100).astype(str).str.cat(sep=' ')
    return bool(emoji_pattern.search(sample_texts))

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def trunc(txt: str, lim: int) -> str:
    return txt if len(txt) <= lim else txt[:lim] + " …[truncated]"

def build_batch_payload(rows: List[Dict[str, Any]]) -> str:
    items = [{"id": str(r["id"]), "text": trunc(r["text"], MAX_TEXT_CHARS)} for r in rows]
    return json.dumps({"items": items}, ensure_ascii=False, indent=2)

def extract_json_from_response(raw_content: str) -> Dict[str, Any]:
    """Extract JSON from response that might contain markdown formatting or extra text."""
    # Remove markdown code blocks if present
    content = re.sub(r'```json\s*', '', raw_content)
    content = re.sub(r'```\s*$', '', content)
    
    # Try to find JSON object in the response
    json_start = content.find('{')
    json_end = content.rfind('}') + 1
    
    if json_start >= 0 and json_end > json_start:
        json_str = content[json_start:json_end]
        return json.loads(json_str)
    else:
        # If no clear JSON structure found, try parsing the whole content
        return json.loads(content.strip())

def call_gpt_batch(client: AzureOpenAI, items_json: str) -> Dict[str, Any]:
    """Updated function without structured output for newer API versions."""
    res = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        temperature=1,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(items_json=items_json)},
        ],
        max_completion_tokens=4000  # Ensure enough tokens for response
    )
    
    raw = res.choices[0].message.content
    print(f"\n---- RAW COMPLETION (first 500 chars) ----\n{raw[:500]}\n" + "-"*50 + "\n")
    
    try:
        return extract_json_from_response(raw)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed, raw content: {raw}")
        raise e

# ────────────────────────── Core batch processing ─────────────────────────── #
def run_batches(df: pd.DataFrame) -> List[Dict[str, Any]]:
    client = get_client()
    preds: List[Dict[str, Any]] = []
    rows = df.to_dict(orient="records")
    
    print(f"Processing {len(rows)} rows in batches of {BATCH_SIZE}...")

    for i, batch in enumerate(chunk(rows, BATCH_SIZE)):
        print(f"Processing batch {i+1} ({len(batch)} items)...")
        items_json = build_batch_payload(batch)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                payload = call_gpt_batch(client, items_json)
                parsed = BatchOut(**payload)  # Validate using Pydantic
                batch_preds = [{"id": it.id, "labels": it.labels} for it in parsed.items]
                preds.extend(batch_preds)
                print(f"✓ Batch {i+1} completed successfully")
                break
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"✗ Batch {i+1} attempt {attempt} failed: {e}")
                
                if attempt < MAX_RETRIES and len(batch) > 1:
                    print(f"  Splitting batch {i+1} and retrying...")
                    mid = len(batch) // 2
                    preds.extend(run_batches(pd.DataFrame(batch[:mid])))
                    preds.extend(run_batches(pd.DataFrame(batch[mid:])))
                    break
                elif attempt < MAX_RETRIES:
                    wait_time = 0.8 * attempt
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  All attempts failed for batch {i+1}, assigning empty labels")
                    preds.extend({"id": str(r["id"]), "labels": []} for r in batch)
                    break
    return preds

# ────────────────────────────── CSV output ───────────────────────────────── #
def save_predictions_csv(predictions: List[Dict[str, Any]], output_file: str) -> None:
    """Save predictions to CSV file with proper formatting."""
    print(f"Saving {len(predictions)} predictions to {output_file}...")
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["id", "labels", "num_labels"])
        
        # Write data
        for pred in predictions:
            labels_str = ";".join(pred["labels"]) if pred["labels"] else ""
            num_labels = len(pred["labels"])
            writer.writerow([pred["id"], labels_str, num_labels])
    
    print(f"✓ Predictions saved to {output_file}")

# ────────────────────────────── CLI wrapper ───────────────────────────────── #
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv> [output_csv]")
        print("Example: python script.py dataTest.csv predictions.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"

    # Validate input file exists
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        sys.exit(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")

    # Validate required columns
    required = {"id", "text"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Input CSV must have columns {required}. Missing: {missing}")

    print(f"Loaded {len(df)} rows from {input_file}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data:")
    print(df.head(2).to_string())
    print("-" * 60)
    # Auto-detect emoji usage and select appropriate prompt
    has_emojis = contains_emojis(df)
    global SYSTEM_MSG
    SYSTEM_MSG = SYSTEM_MSG_WITH_EMOJIS if has_emojis else SYSTEM_MSG_WITHOUT_EMOJIS
    print(f"Using {'emoji' if has_emojis else 'no-emoji'} prompt version")
    print("-" * 60)

    # Process predictions
    try:
        predictions = run_batches(df[["id", "text"]].copy())
        
        # Save to CSV
        save_predictions_csv(predictions, output_file)
        
        # Summary statistics
        total_labels = sum(len(p["labels"]) for p in predictions)
        avg_labels = total_labels / len(predictions) if predictions else 0
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Total rows processed: {len(predictions)}")
        print(f"  Total labels assigned: {total_labels}")
        print(f"  Average labels per text: {avg_labels:.2f}")
        print(f"  Output saved to: {output_file}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()