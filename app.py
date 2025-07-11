from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from serpapi import GoogleSearch
import json
from selenium_context import create_driver 
import os
import gc

from utils import (
    model,
    is_pdf,
    is_webpage,
    scrape_with_selenium,
    split_text,
    gemini_model_summary,
    join_summaries,
    collect_valid_images_from_links,
    convert_images_to_llm_strings,
    chunk_descriptions,
    evaluate_chunks_with_llm,
    filter_top_images,
    generate_carousel_prompt,
    replace_image_indexes_with_urls,
    serp_api_key,
    max_length  # assuming this is also in utils
)

app = Flask(__name__)

def scrape_page_isolated(url):
    with create_driver() as driver:
        return scrape_with_selenium(url, driver)

@app.route('/generate_content', methods=['POST'])
def generate_content():
    try:
        data = request.get_json()

        topic = data.get('topic', 'GPT prompting ways')
        links_to_be_search = data.get('links_to_be_search', [])
        Extra_Content_Description = data.get('Extra_Content_Description', '')
        Content_Specifications = data.get('Content_Specifications', [])
        json_template_str = data.get('json_template_str', None)

        # SERP API search
        params = {
            'q': topic,
            'api_key': serp_api_key,
            'engine': 'google',
            'num': '5',
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        results_list = []
        for result in results.get('organic_results', []):
            title = result.get('title', 'No Title')
            link = result.get('link', '')
            if link:
                results_list.append({'title': title, 'link': link})
                
        serp_links = {result.get('link') for result in results.get('organic_results', []) if result.get('link')}
        links_to_be_search = [link for link in links_to_be_search if link not in serp_links]
        del serp_links 
        gc.collect()

        for link in links_to_be_search:
            results_list.append({'title': 'Custom Link', 'link': link})

        results_list_updated = []
        links_only = []

        for result in results_list:
            url = result['link']
            if not url or is_pdf(url) or not is_webpage(url):
                continue

            title, content = scrape_page_isolated(url)
            if content:
                links_only.append(url)
                chunks = split_text(content)
                for chunk in chunks:
                    results_list_updated.append({
                        'title': title,
                        'content': chunk
                    })
                del chunks
                gc.collect()
        del results_list
        gc.collect()

        summaries = []
        for result in results_list_updated:
            title = result.get('title', '')
            content = result.get('content', '')
            summary_result = gemini_model_summary(title, content, topic)
            summaries.append({
                'title': title,
                'summary': summary_result.get("summary", ""),
                'content_index': summary_result.get("content_index", 0)
            })
        del results_list_updated
        gc.collect()

        summary_text = join_summaries(summaries)
        del summaries
        gc.collect()
        with create_driver() as driver:
            final_images = collect_valid_images_from_links(links_only, results_list, driver)
            final_images = final_images[:20]

        # final_images = collect_valid_images_from_links(links_only, results_list,driver)
        del links_only
        gc.collect()
        desc_strings, img_links = convert_images_to_llm_strings(final_images)

        chunks = chunk_descriptions(desc_strings)
        evaluated = evaluate_chunks_with_llm(chunks,topic)
        del desc_strings  
        del chunks 
        gc.collect()
        top_images = filter_top_images(evaluated, img_links)

        prompt = generate_carousel_prompt(
            topic=topic,
            summary_text=summary_text,
            top_images = top_images,
            json_template_str = json_template_str,
            Extra_Content_Description = Extra_Content_Description,
            Content_Specifications = Content_Specifications
        )
        del summary_text
        del json_template_str
        del Extra_Content_Description
        del Content_Specifications
        gc.collect()

        response = model.generate_content(prompt)
        del prompt
        gc.collect()
        raw_output = response.text.strip()

        if raw_output.startswith("```json"):
            raw_output = raw_output.lstrip("```json").rstrip("```").strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output.lstrip("```").rstrip("```").strip()

        try:
            carousel_json = json.loads(raw_output)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            carousel_json = None

        final_content = replace_image_indexes_with_urls(carousel_json,img_links)

        return jsonify({
            "status": "success",
            "final_content": final_content,
            "top_images": top_images,
            "image_sources": img_links
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
        
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))  # Default is optional, e.g., 10000 for local testing
    app.run(host='0.0.0.0', port=port)
