from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.pipeline.prediction import PredictionPipeline


app = FastAPI(
    title="Legal Text Summarizer",
    description="Generate summaries of legal documents using BART.",
    version="1.0.0",
)

prediction_pipeline = None


class SummarizeRequest(BaseModel):
    legal_text: str


def get_prediction_pipeline():
    global prediction_pipeline

    if prediction_pipeline is None:
        config = ConfigurationManager().get_prediction_config()
        prediction_pipeline = PredictionPipeline(config)

    return prediction_pipeline


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Legal Text Summarizer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                background: #f4f6f8;
                color: #222;
            }
            textarea {
                width: 100%;
                min-height: 250px;
                padding: 12px;
                box-sizing: border-box;
                font-size: 15px;
                border: 1px solid #bbb;
                border-radius: 8px;
            }
            button {
                margin-top: 12px;
                padding: 12px 20px;
                background: #155eef;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }
            button:disabled {
                opacity: 0.6;
                cursor: wait;
            }
            #summary {
                white-space: pre-wrap;
                background: white;
                padding: 16px;
                border-radius: 8px;
                line-height: 1.6;
            }
            .panel {
                margin-top: 24px;
            }
            #status {
                color: #555;
            }
        </style>
    </head>
    <body>
        <h1>Legal Text Summarizer</h1>
        <p>Paste a legal document below to generate a concise summary.</p>

        <textarea id="legalText"
            placeholder="Paste your legal document here..."></textarea>

        <button id="summarizeButton" onclick="summarize()">
            Generate Summary
        </button>

        <p id="status"></p>

        <div class="panel">
            <h2>Generated Summary</h2>
            <div id="summary">Your summary will appear here.</div>
        </div>

        <script>
            async function summarize() {
                const legalText =
                    document.getElementById("legalText").value.trim();
                const button =
                    document.getElementById("summarizeButton");
                const status = document.getElementById("status");
                const output = document.getElementById("summary");

                if (!legalText) {
                    status.textContent = "Please enter a legal document.";
                    return;
                }

                button.disabled = true;
                status.textContent = "Generating summary...";
                output.textContent = "";

                try {
                    const response = await fetch("/summarize", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            legal_text: legalText
                        })
                    });

                    const result = await response.json();

                    if (!response.ok) {
                        throw new Error(
                            result.detail || "Summarization failed."
                        );
                    }

                    output.textContent = result.summary;
                    status.textContent = "Summary generated successfully.";
                } catch (error) {
                    status.textContent = "Error: " + error.message;
                } finally {
                    button.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/summarize")
def summarize(request: SummarizeRequest):
    if not request.legal_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide a non-empty legal document.",
        )

    try:
        pipeline = get_prediction_pipeline()
        summary = pipeline.predict(request.legal_text)

        return {"summary": summary}

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The fine-tuned BART model or tokenizer was not found. "
                "Check the paths in config/config.yaml."
            ),
        ) from error