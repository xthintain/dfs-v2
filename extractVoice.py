from openai import OpenAI
import json
import time
import re

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

def call_llama_api(words):
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You must generate JSON strictly following this template:

{
  "storyboard": [
    {
      "scene_id": 1,
      "shot_type": "wide",
      "duration_sec": 5,
      "description": "scene description here",
      "prompt": "high quality, detailed prompt here",
      "negative_prompt": "low quality, blurry, deformed", 
      "camera_params": "24mm, f/2.8"
    },
    {
      "scene_id": 2,
      "shot_type": "medium", 
      "duration_sec": 3,
      "description": "another scene description",
      "prompt": "another detailed prompt",
      "negative_prompt": "low quality, blurry",
      "camera_params": "50mm, f/1.8"
    }
    // Continue with exactly 20 scenes like this
  ]
}

Rules you MUST follow:
1. Use strict JSON format - all quotes, commas and brackets must be correct
2. Include exactly 20 scenes
3. Each scene must have all specified fields
4. Description must match the input text
5. Prompts should be detailed and in English

Example high-quality prompt:
"high quality, 8k, ultra detailed, a young Asian scientist in lab coat, futuristic lab with holograms, soft lighting, cinematic style, 85mm lens"


"""
    prompt += f"""
    \n
    Now generate storyboards for this text:
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {words}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>        
"""

    try:
        max_retries = 5
        retry_delay = 1
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print("="*50)
                print(f"Attempt {attempt+1}/{max_retries}")
                print(f"Prompt length: {len(prompt)}")
                print("="*50)

                response = client.chat.completions.create(
                    model="meta-llama/llama-3.3-70b-instruct:free",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=18384,
                    timeout=30
                )
                
                if not response or not response.choices:
                    raise ValueError("Empty API response")
                    
                content = response.choices[0].message.content
                
                print("="*50)
                print("Raw API Response:")
                print(content)
                print("="*50)
                
                # Try multiple methods to extract JSON
                json_str = None
                methods = [
                    # Try finding complete JSON block
                    lambda: re.search(r'\{[\s\S]*\}', content).group(0),
                    # Try finding just the storyboard array
                    lambda: '{"storyboard":' + re.search(r'\[[\s\S]*\]', content).group(0) + '}',
                    # Try extracting between ```json ``` markers
                    lambda: re.search(r'```json\n([\s\S]*?)\n```', content).group(1)
                ]
                
                for method in methods:
                    try:
                        json_str = method()
                        break
                    except:
                        continue
                        
                if not json_str:
                    raise ValueError("Could not extract JSON from response")
                
                # Try parsing directly first
                try:
                    json_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Initial parse failed, attempting fixes... Error: {str(e)}")
                
                # Apply progressive fixes
                fixes = [
                    # Remove code block markers
                    lambda s: re.sub(r'^```(json)?\n|\n```$', '', s, flags=re.IGNORECASE),
                    # Add missing commas between objects
                    lambda s: re.sub(r'}\s*{', '},{', s),
                    # Add missing quotes around values
                    lambda s: re.sub(r'"\s*:\s*([^"\s{][^,\n}]*)', r'": "\1"', s),
                    # Ensure proper array commas
                    lambda s: re.sub(r'"\s*(?=")', r'",', s),
                    # Ensure closing brace
                    lambda s: s + '}' if not s.strip().endswith('}') else s
                ]
                
                for fix in fixes:
                    try:
                        json_str = fix(json_str)
                        json_data = json.loads(json_str)
                        print("Successfully parsed after fixes")
                        break
                    except:
                        continue
                else:
                    raise ValueError("Could not parse JSON after all fixes")
                
                if not isinstance(json_data.get('storyboard'), list):
                    raise ValueError("Missing storyboard array")
                
                # Add position info
                for scene in json_data['storyboard']:
                    desc = scene['description']
                    start = words.find(desc)
                    if start >= 0:
                        scene['start_pos'] = start
                        scene['end_pos'] = start + len(desc)
                    else:
                        scene['start_pos'] = -1
                        scene['end_pos'] = -1
                
                return json_data
                
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                
        raise ValueError(f"API call failed after {max_retries} attempts: {str(last_error)}")
            
    except Exception as e:
        print("="*50)
        print("Error details:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("="*50)
        raise

# Example usage
if __name__ == "__main__":
    test_text = "Sample story text here..."
    result = call_llama_api(test_text)
    print(result)
