"""
Gradio demo app — run with: python app.py
"""

import gradio as gr
from src.model.predict import review_code

SAMPLE_CODE = '''def read_config(path):
    f = open(path)
    data = f.read()
    return data

def get_user(users, idx):
    for i in range(len(users) + 1):
        print(users[i])

def process(obj):
    value = obj.get_value()
    return value
'''

def run_review(code, language):
    if not code.strip():
        return "Please enter some code."
    try:
        findings = review_code(code, language.lower())
    except Exception as e:
        return f"Error: {str(e)}"

    output = []
    for f in findings:
        icon = "🔴" if f["severity"] == "high" else ("🟡" if f["severity"] == "medium" else "🟢")
        output.append(f"{icon} **{f['function_name']}** (lines {f['start_line']}–{f['end_line']})")
        output.append(f"   Bug type : `{f['bug_type']}`")
        output.append(f"   Severity : {f['severity'].upper()}")
        output.append(f"   Confidence: {f['confidence']*100:.1f}%")
        output.append(f"   Fix hint : {f['suggestion']}")
        output.append("")

    return "\n".join(output) if output else "No findings."


with gr.Blocks(title="AI Code Reviewer") as demo:
    gr.Markdown("# 🔍 AI Code Reviewer\nDetects bugs using fine-tuned CodeBERT")

    with gr.Row():
        with gr.Column():
            code_input = gr.Textbox(label="Paste your code here",
                                     value=SAMPLE_CODE, lines=20)
            lang_input = gr.Dropdown(["python", "javascript"],
                                      label="Language", value="python")
            submit_btn = gr.Button("Review Code", variant="primary")
        with gr.Column():
            output_box = gr.Markdown(label="Review Results")

    submit_btn.click(fn=run_review,
                     inputs=[code_input, lang_input],
                     outputs=output_box)

if __name__ == "__main__":
    demo.launch(share=True)