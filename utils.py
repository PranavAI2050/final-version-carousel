import google.generativeai as genai
from google.generativeai import GenerativeModel
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import mimetypes
import re
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import os

load_dotenv()
serp_api_key = os.environ.get("serp_api_key")
gemini_key = os.environ.get("gemini_key")

genai.configure(api_key= gemini_key)
model = GenerativeModel('gemini-2.0-flash-lite')
max_length = 100000  
MAX_CHARS = 100000
MIN_SCORE = 6 


def join_summaries(summaries, threshold=6):
    return " ".join([
        summary['summary'].replace("\n", " ").replace("\r", " ")
        for summary in summaries
        if summary.get('content_index', 0) > threshold
    ])

def is_webpage(url):
    try:
        print(f"[DEBUG] Checking if URL is a webpage: {url}")
        response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = response.headers.get('Content-Type', '').lower()
        print(f"[DEBUG] Content-Type for {url}: {content_type}")
        return 'text/html' in content_type
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to check if URL is a webpage ({url}): {e}")
        return False


def is_pdf(url):
    try:
        print(f"[DEBUG] Checking if URL is a PDF: {url}")
        response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = response.headers.get('Content-Type', '').lower()
        print(f"[DEBUG] Content-Type for {url}: {content_type}")
        return 'application/pdf' in content_type
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to check if URL is a PDF ({url}): {e}")
        return False


def clean_text(text):
    print(f"[DEBUG] Cleaning text of length: {len(text)}")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_text(text, max_length=max_length):
    print(f"[DEBUG] Splitting text of length {len(text)} into chunks of max {max_length}")
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]


def scrape_with_selenium(url, driver):
    try:
        print(f"[INFO] Scraping URL with Selenium: {url}")
        driver.get(url)
        time.sleep(3)  # You can replace this with WebDriverWait for better control

        soup = BeautifulSoup(driver.page_source, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "No Title"

        print(f"[DEBUG] Page title: {title}")

        # Scrape visible text from body
        body_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
        print(f"[DEBUG] Scraped body text length: {len(body_text)}")
        body_text = clean_text(body_text)

        return title, body_text
    except Exception as e:
        print(f"[ERROR] Error scraping {url}: {e}")
        return "No Title", ""


def gemini_model_summary(title, content, topic):
    print(f"[DEBUG] Starting summary generation for title: {title[:60]}...")  # Log a snippet of the title
    if not content.strip():
        print("[WARN] Empty content received for summarization.")
        return {"summary": "", "content_index": 0}

    prompt = f"""
You are an AI assistant analyzing an article to assess its relevance to a given topic and generate a clear, concise summary.

Your task:
1. Write a **concise summary** (2–4 sentences max) highlighting:
   - Key facts
   - Named entities
   - Relevant quotes or statistics
   - Information suitable for a social media carousel
2. Provide a **content_index** score between 0 and 10:
   - 0 = not relevant to the topic
   - 10 = highly relevant for carousel content
   - 1–9 = partially relevant

Only respond in valid JSON format, like below:

{{
  "summary": "Short, clear summary here.",
  "content_index": 7
}}

---
Topic: {topic}
Title: {title}

[CONTENT START]
{content}
[CONTENT END]
"""

    try:
        print(f"[DEBUG] Sending prompt to model for topic: {topic}")
        response = model.generate_content(prompt)
        text = response.text.strip()
        print(f"[DEBUG] Raw model response: {text[:120]}...")  # Log beginning of response

        # Remove any code block formatting like ```json or ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        parsed = json.loads(text)
        print(f"[DEBUG] Parsed JSON response: {parsed}")

        if not isinstance(parsed, dict) or 'summary' not in parsed or 'content_index' not in parsed:
            raise ValueError("Invalid response format")

        return parsed

    except Exception as e:
        print(f"[ERROR] Failed to generate or parse summary: {e}")
        return {"summary": f"Error generating summary: {e}", "content_index": 0}


def extract_images_with_context(url, driver, valid_extensions=('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')):
    print(f"[DEBUG] Navigating to URL: {url}")
    driver.get(url)
    time.sleep(3)  # Allow JS to load fully

    print("[DEBUG] Parsing page source with BeautifulSoup...")
    soup = BeautifulSoup(driver.page_source, "html.parser")
    images_data = []

    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""
    print(f"[DEBUG] Page title extracted: '{page_title}'")

    # 1. Open Graph image
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        og_src = og_image["content"].strip()
        print(f"[DEBUG] Found Open Graph image: {og_src}")
        images_data.append({
            "title": page_title,
            "src": og_src,
            "alt": "Open Graph Image",
            "context": "This is the Open Graph image used for previews",
            "width": None,
            "height": None,
            "position_index": -1,
            "is_probably_logo": False,
            "is_og_image": True
        })

    print("[DEBUG] Finding all <img> tags...")
    all_imgs = driver.find_elements("tag name", "img")
    bs_imgs = soup.find_all("img")
    print(f"[DEBUG] Total Selenium images: {len(all_imgs)}, BeautifulSoup images: {len(bs_imgs)}")

    for idx, img_el in enumerate(all_imgs):
        try:
            if idx >= len(bs_imgs):
                print(f"[WARN] Skipping index {idx} due to BS4 mismatch")
                continue

            img_tag = bs_imgs[idx]
            alt_text = img_tag.get("alt", "").strip()
            src = img_tag.get("src", "") or img_tag.get("data-src", "")
            src = src.strip()

            if not src or not src.lower().endswith(valid_extensions):
                print(f"[DEBUG] Skipping image {idx} due to invalid src or extension: {src}")
                continue

            if re.search(r"(sprite|icon|logo|tracking|ads|analytics)", src.lower()):
                print(f"[DEBUG] Skipping image {idx} due to unwanted pattern: {src}")
                continue

            width = img_el.size.get("width", 0)
            height = img_el.size.get("height", 0)
            if width < 50 or height < 50:
                print(f"[DEBUG] Skipping image {idx} due to small size: {width}x{height}")
                continue

            context = ""

            figure = img_tag.find_parent("figure")
            if figure:
                figcaption = figure.find("figcaption")
                if figcaption:
                    context = figcaption.get_text(strip=True)
                    print(f"[DEBUG] Found figcaption for image {idx}: {context}")

            if not context:
                current = img_tag
                for _ in range(2):
                    parent = current.find_parent()
                    if parent and parent.get_text(strip=True):
                        context = parent.get_text(strip=True)
                        print(f"[DEBUG] Found parent context for image {idx}")
                        break
                    current = parent if parent else current

            if not context:
                prev = img_tag.find_previous_sibling(["p", "h2", "h3"])
                if prev:
                    context = prev.get_text(strip=True)
                    print(f"[DEBUG] Found previous sibling context for image {idx}")

            images_data.append({
                "title": page_title,
                "src": src,
                "alt": alt_text,
                "context": context,
                "width": width,
                "height": height,
                "position_index": idx,
                "is_probably_logo": "logo" in src.lower() or "logo" in alt_text.lower(),
                "is_og_image": False
            })
            print(f"[INFO] Image {idx} added: {src}")

        except Exception as e:
            print(f"[ERROR] Image {idx} skipped due to error: {e}")
            continue

    print(f"[DEBUG] Total valid images extracted: {len(images_data)}")
    return images_data


def collect_valid_images_from_links(links, results_list, driver):
    print(f"[DEBUG] Starting image collection from {len(links)} links...")

    all_valid_images = []

    for i, url in enumerate(links):
        title = results_list[i].get('title', f"Page {i+1}")
        print(f"🔗 [INFO] Processing: '{title}' ({url})")

        try:
            images = extract_images_with_context(url, driver)
            print(f"[DEBUG] Found {len(images)} images from: {url}")
            all_valid_images.extend(images)
        except Exception as e:
            print(f"❌ [ERROR] Failed to process '{url}': {e}")

    print(f"[DEBUG] Total valid images collected: {len(all_valid_images)}")
    return all_valid_images


def convert_images_to_llm_strings(images_data):
    print(f"[DEBUG] Converting {len(images_data)} images to LLM strings...")

    description_strings = []
    image_urls = []

    for idx, img in enumerate(images_data):
        src = img.get('src', '')
        if not src:
            print(f"[WARN] Image {idx} missing 'src', skipping...")
            continue

        description = (
            f"[{idx}] Title: {img.get('title', 'Unknown')}\n"
            f"Alt text: {img.get('alt', 'None')}\n"
            f"Context: {img.get('context', 'None')}\n"
            f"Dimensions: {img.get('width')}x{img.get('height')}\n"
            f"Likely Logo: {'Yes' if img.get('is_probably_logo') else 'No'}\n"
            f"Open Graph Image: {'Yes' if img.get('is_og_image') else 'No'}"
        )

        print(f"[INFO] Image {idx} processed with description.")
        description_strings.append(description)
        image_urls.append(src)

    print(f"[DEBUG] Generated {len(description_strings)} descriptions and URLs.")
    return description_strings, image_urls


def chunk_descriptions(desc_strings, max_chars=MAX_CHARS):
    """Split image descriptions into chunks under max_chars."""
    print(f"[DEBUG] Starting chunking of {len(desc_strings)} descriptions with max_chars={max_chars}")

    chunks, current_chunk, current_len = [], [], 0

    for i, desc in enumerate(desc_strings):
        desc_len = len(desc)
        print(f"[DEBUG] Processing description {i} (length={desc_len})")

        if current_len + desc_len > max_chars and current_chunk:
            print(f"[INFO] Chunk limit reached. Finalizing chunk with {len(current_chunk)} descriptions.")
            chunks.append(current_chunk)
            current_chunk, current_len = [], 0

        current_chunk.append(desc)
        current_len += desc_len

    if current_chunk:
        print(f"[INFO] Adding final chunk with {len(current_chunk)} descriptions.")
        chunks.append(current_chunk)

    print(f"[DEBUG] Total chunks created: {len(chunks)}")
    return chunks


def build_prompt(chunk, topic):
    """Builds the LLM prompt for a chunk of image descriptions."""
    print(f"[DEBUG] Building prompt for chunk with {len(chunk)} image descriptions on topic: '{topic}'")

    image_descriptions = "\n\n".join(chunk)

    prompt = f"""
You are a helpful AI that evaluates a series of image descriptions to determine their usefulness for creating visual carousels related to topic: {topic}.

For each image, return:
- "index": the image number (as provided),
- "score": an integer 1–10 showing how relevant/valuable the image is,
- "reason": why this image may or may not be useful for content generation.

Only respond in JSON list format.

Images:
{image_descriptions}

Respond like:
[
  {{
    "index": 0,
    "score": 9,
    "reason": "Shows core idea with relevant caption"
  }},
  ...
]
"""
    print("[DEBUG] Prompt successfully built.")
    return prompt


def evaluate_chunks_with_llm(chunks, topic):
    print(f"[DEBUG] Starting LLM evaluation for {len(chunks)} chunks on topic: '{topic}'")
    all_results = []

    for i, chunk in enumerate(chunks):
        print(f"[INFO] Evaluating chunk {i+1}/{len(chunks)}")
        prompt = build_prompt(chunk, topic)

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            # Clean markdown-style code block formatting
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            parsed = json.loads(text)
            print(f"[DEBUG] Chunk {i+1}: Parsed {len(parsed)} evaluation results")

            for result in parsed:
                if isinstance(result, dict) and "index" in result:
                    all_results.append(result)
                else:
                    print(f"[WARNING] Invalid result format in chunk {i+1}: {result}")

        except Exception as e:
            print(f"[ERROR] Error evaluating chunk {i+1}: {e}")

    print(f"[DEBUG] Total valid evaluation results collected: {len(all_results)}")
    return all_results


def filter_top_images(evaluated_results, img_links, threshold=MIN_SCORE):
    print(f"[DEBUG] Filtering top images from {len(evaluated_results)} evaluated results using threshold={threshold}")
    selected = []

    for item in evaluated_results:
        index = item.get("index")
        score = item.get("score", 0)
        reason = item.get("reason", "")

        if score >= threshold and index is not None and index < len(img_links):
            print(f"[INFO] ✅ Image {index} selected (Score: {score}) – {reason}")
            selected.append({
                "index": index,
                "score": score,
                "reason": reason,
            })
        else:
            print(f"[DEBUG] ❌ Image {index} skipped (Score: {score})")

    print(f"[DEBUG] Total selected top images: {len(selected)}")
    return selected


def generate_carousel_prompt(topic, summary_text, top_images, json_template_str, Extra_Content_Description, Content_Specifications):
    import json

    print("[DEBUG] Generating carousel prompt...")
    print(f"[INFO] Topic: {topic}")
    print(f"[INFO] Summary length: {len(summary_text)} characters")
    print(f"[INFO] Number of top images: {len(top_images)}")
    print(f"[INFO] Content specifications provided: {bool(Content_Specifications)}")
    print(f"[INFO] Extra content description provided: {bool(Extra_Content_Description.strip())}")

    # Ensure Content_Specifications is a list if None is passed
    if Content_Specifications is None:
        Content_Specifications = []

    # Step 1: Format top_images for display
    formatted_images = "\n".join([
        f"[{img['index']}] Score: {img['score']} – {img['reason']}"
        for img in top_images
    ])
    print("[DEBUG] Formatted image entries:")
    print(formatted_images)

    # Step 2: Serialize the JSON template
    json_template_str = json.dumps(json_template_str, indent=2)
    print("[DEBUG] JSON template string serialized successfully.")

    # Step 3: Optional content sections
    extra_content_section = f"\nAdditional creative guidance:\n{Extra_Content_Description.strip()}\n" if Extra_Content_Description.strip() else ""
    content_spec_section = "\nSpecific content constraints and requirements:\n" + "\n".join([f"- {spec}" for spec in Content_Specifications]) if Content_Specifications else ""
    optional_guidance = (extra_content_section + content_spec_section).strip()

    if optional_guidance:
        print("[DEBUG] Optional guidance included in prompt.")
        optional_guidance = f"\n---\n\n## Additional Notes\n{optional_guidance}"
    else:
        print("[DEBUG] No optional guidance provided.")

    # Step 4: Construct the prompt
    prompt = f"""
You are an AI assistant helping to design visually engaging carousel slides for social media.

## Objective:
Fill in the following JSON template with content for a carousel about the topic: **"{topic}"**.

Use the summarized content and image options below to generate text and select suitable images.

---

### Rules for Filling the Template:

1. If `"content": "text"`:
   - Write **concise, relevant content** based on the summary and the element's `"description"`.
   - If a character or word limit is mentioned, **strictly follow it**.
   - The tone should match the title or description (e.g., persuasive, informative, punchy, etc.).

2. If `"content": "url"`:
   - Replace `"url"` with an image reference using this format:
     `"image:index"`
     Example: `"image:7"` = image at index 7.
   - **Only use the images from the provided list of `top_images`**.
   - Do **not invent or guess image content**.
   - Pick the best image based on its score and explanation (`reason`).

---

## Topic
{topic}

---

## Summarized Content
\"\"\" 
{summary_text}
\"\"\" 
{optional_guidance}

---

## Top Images to Choose From
Each entry includes an index, a relevance score (1–10), and a reason explaining its usefulness:

{formatted_images}

---

## JSON Template
Replace only the `"content"` values. Do not change any other keys or structure.

{json_template_str}

---

## Expected Output
Return ONLY the completed JSON object, with each `"content"` field correctly filled in.

- Use `"image:index"` for images (e.g., `"image:0"`).
- Follow all text formatting rules and word limits.
"""

    print("[DEBUG] Prompt generation complete. Prompt length:", len(prompt), "characters.")
    return prompt


def replace_image_indexes_with_urls(llm_output_json, img_links):
    """
    Replace 'image:X' placeholders in LLM output with actual image URLs from img_links list.
    """
    import copy

    print("[DEBUG] Starting image URL replacement...")
    print(f"[INFO] Number of image links available: {len(img_links)}")

    def replace_content(node, path="root"):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = f"{path}.{key}"
                if isinstance(value, dict):
                    replace_content(value, new_path)
                elif isinstance(value, str) and value.startswith("image:"):
                    try:
                        index = int(value.split(":")[1])
                        if 0 <= index < len(img_links):
                            print(f"[REPLACE] At {new_path} - Replacing 'image:{index}' with URL: {img_links[index]}")
                            node[key] = img_links[index]
                        else:
                            print(f"[WARNING] Index out of range at {new_path}: {index}")
                    except (IndexError, ValueError) as e:
                        print(f"[ERROR] Failed to parse image index at {new_path}: {value} – {e}")
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                replace_content(item, f"{path}[{idx}]")

    updated = copy.deepcopy(llm_output_json)
    replace_content(updated)

    print("[DEBUG] Image replacement complete.")
    return updated
