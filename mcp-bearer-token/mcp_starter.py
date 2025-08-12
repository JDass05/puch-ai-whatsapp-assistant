import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
from bs4 import BeautifulSoup  # add this import at the top if not present


import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]
    
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/v1/chat/completions")

async def call_ollama(prompt: str) -> str:
    import httpx

    payload = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.2,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(OLLAMA_API_URL, json=payload, timeout=30)
            response.raise_for_status()
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Ollama API error: {e}"))

        data = response.json()
        # Adjust if your Ollama response format differs
        return data["choices"][0]["message"]["content"]


# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)


@mcp.tool
async def about() -> dict:
    return {"name": mcp.name, "description": "An Example Bearer token mcp server"}


# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# # --- Tool: job_finder (now smart!) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# # Image inputs and sending images

# MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
#     description="Convert an image to black and white and save it.",
#     use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
#     side_effects="The image will be processed and saved in a black and white format.",
# )



# @mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
# async def make_img_black_and_white(
#     puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
# ) -> list[TextContent | ImageContent]:
#     import base64
#     import io

#     from PIL import Image

#     try:
#         image_bytes = base64.b64decode(puch_image_data)
#         image = Image.open(io.BytesIO(image_bytes))

#         bw_image = image.convert("L")

#         buf = io.BytesIO()
#         bw_image.save(buf, format="PNG")
#         bw_bytes = buf.getvalue()
#         bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

#         return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# @mcp.tool(
#     description=RichToolDescription(
#         description="Fact check or analyze jobs using local Ollama LLaMA 3.2 with optional web search.",
#         use_when="Use this tool to get deep job or topic analysis with web context and local LLM reasoning.",
#         side_effects="Calls local Ollama LLaMA server; may fetch and scrape web URLs.",
#     ).model_dump_json()
# )
# async def job_finder_ollama(
#     user_goal: Annotated[str, Field(description="User's goal or freeform query")],
#     job_description: Annotated[str | None, Field(description="Job description text")] = None,
#     job_url: Annotated[AnyUrl | None, Field(description="URL to job posting")] = None,
#     raw: Annotated[bool, Field(default=False, description="Return raw HTML if True")] = False,
# ) -> str:
#     prompt = f"User goal:\n{user_goal}\n\n"

#     # Add job description if available
#     if job_description:
#         prompt += f"Job Description:\n{job_description}\n\n"

#     # Add content from URL if available
#     if job_url:
#         content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
#         prompt += f"Job Posting URL Content:\n{content}\n\n"

#     # If no description or URL, do a web search and fetch top 3 results
#     if not job_description and not job_url:
#         # Simple DuckDuckGo search for user_goal
#         ddg_url = f"https://html.duckduckgo.com/html/?q={user_goal.replace(' ', '+')}"
#         async with httpx.AsyncClient() as client:
#             resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
#             if resp.status_code == 200:
#                 soup = BeautifulSoup(resp.text, "html.parser")
#                 links = []
#                 for a in soup.find_all("a", class_="result__a", href=True):
#                     href = a["href"]
#                     if href.startswith("http"):
#                         links.append(href)
#                     if len(links) >= 3:
#                         break
#                 prompt += "Related links and their content snippets:\n"
#                 for i, url in enumerate(links):
#                     try:
#                         content_md = await Fetch.fetch_url(url)
#                         snippet = content_md[:1500]  # truncate to 1500 chars
#                         prompt += f"\nSource {i+1} ({url}):\n{snippet}\n"
#                     except McpError:
#                         prompt += f"\nSource {i+1} ({url}): <Failed to fetch content>\n"
#             else:
#                 prompt += "\n<Failed to perform web search to augment input>\n"

#     # Final prompt addition
#     prompt += "\nPlease analyze the above and provide a detailed response."

#     # Call Ollama with combined prompt
#     result = await a(prompt)

#     return result
# --- Run MCP Server ---

# 1. Fact Checker
@mcp.tool(description=RichToolDescription(
    description="Verify the truthfulness of a statement based on provided evidence.",
    use_when="Use this to fact-check any claim with supporting context.",
    side_effects="Returns a fact-check report indicating true, false, or uncertain."
).model_dump_json())
async def fact_checker_online(
    statement: Annotated[str, Field(description="Statement to verify")],
    context_text: Annotated[str | None, Field(description="Supporting text or webpage content to check against")] = None,
) -> str:
    prompt = f"""
You are an expert fact-checker AI assistant.

Your task: Verify the truthfulness of the following statement using all available evidence provided. Carefully analyze the context and sources, considering possible nuances or uncertainties.

Statement:
\"\"\"{statement}\"\"\"

Supporting source information extracted from webpages (may contain summaries or raw content):
\"\"\"{context_text or 'No additional context provided.'}\"\"\"

Instructions:
- Clearly determine if the statement is True, False, or Uncertain based on evidence.
- Provide a concise explanation for your conclusion.
- Mention which sources or facts support your verdict.
- Highlight any contradictory evidence if applicable.
- If evidence is insufficient or inconclusive, say so clearly.

Respond as a professional fact-checking report.
"""
    return await call_ollama(prompt)


# 2. Message Tone Checker
@mcp.tool(description=RichToolDescription(
    description="Analyze the emotional and stylistic tone of a message.",
    use_when="Use this to understand the tone and style of any text message.",
    side_effects="Returns tone classification and explanation."
).model_dump_json())
async def message_tone_checker_o(
    message: Annotated[str, Field(description="Message text to analyze tone")],
) -> str:
    prompt = f"""
You are a language expert specializing in analyzing the emotional and stylistic tone of written communication.

Analyze the following message carefully and classify its tone. Consider subtleties such as sarcasm, politeness, formality, urgency, friendliness, and any implied emotions.

Message:
\"\"\"{message}\"\"\"

Instructions:
- Identify the primary tone(s) (e.g., friendly, formal, sarcastic, angry, neutral).
- Provide a brief explanation of how you inferred this tone.
- Suggest how the message might be perceived by a typical reader.
- If multiple tones are present, describe the mix and their interactions.

Present your analysis clearly and concisely.
"""
    return await call_ollama(prompt)


# 3. Message Translator
@mcp.tool(description=RichToolDescription(
    description="Translate a text message accurately into a specified language.",
    use_when="Use this to translate messages preserving tone and meaning.",
    side_effects="Returns natural, fluent translation."
).model_dump_json())
async def message_translator_o(
    text: Annotated[str, Field(description="Text to translate")],
    target_language: Annotated[str, Field(description="Language to translate into, e.g. 'Spanish'")] = "English",
) -> str:
    prompt = f"""
You are a highly skilled translator AI that accurately translates text between languages.

Translate the following text from its original language into {target_language}.

Text to translate:
\"\"\"{text}\"\"\"

Instructions:
- Preserve the original meaning, tone, and style.
- Ensure the translation is natural, fluent, and culturally appropriate.
- Avoid literal word-for-word translation if it reduces readability.
- Provide only the translated text as the output.

Begin the translation below:
"""
    return await call_ollama(prompt)


# 4. Scam Detection
@mcp.tool(description=RichToolDescription(
    description="Detect potential scams or phishing in messages.",
    use_when="Use this to identify scam likelihood and suspicious content.",
    side_effects="Returns scam risk rating and security advice."
).model_dump_json())
async def scam_detector_o(
    message: Annotated[str, Field(description="Message to analyze for scam risk")],
) -> str:
    prompt = f"""
You are a cybersecurity and fraud detection AI specialized in identifying scams, phishing, and fraudulent messages.

Analyze the following message for potential scam or phishing content:

\"\"\"{message}\"\"\"

Instructions:
- Evaluate likelihood on a scale: Low, Medium, High.
- Identify common scam indicators present (e.g., urgency, suspicious links, requests for sensitive info).
- Explain why you assigned the likelihood rating.
- Advise caution if the message is suspicious.
- If no scam signs are present, clearly state the message appears safe.

Provide your analysis as a detailed, professional security advisory.
"""
    return await call_ollama(prompt)
@mcp.tool(description=RichToolDescription(
    description="Answer user queries by fetching and summarizing relevant web pages or search results.",
    use_when="Use this tool to get useful info from web pages or search queries and provide clear, referenced answers.",
    side_effects="Fetches web content and synthesizes answers with source references."
).model_dump_json())
async def whatsapp_search_engine_online(
    query: Annotated[str, Field(description="User question or search query")],
    url: Annotated[AnyUrl | None, Field(description="Optional URL to fetch relevant information from")] = None,
) -> str:
    fetched_texts = []

    if url:
        # Fetch and simplify content from the given URL
        try:
            content, _ = await Fetch.fetch_url(str(url), Fetch.USER_AGENT)
            fetched_texts.append(f"Content from {url}:\n{content.strip()}")
        except Exception as e:
            return f"❌ Failed to fetch or simplify content from URL: {e}"

    else:
        # No URL provided - do a DuckDuckGo search and fetch top results content
        try:
            links = await Fetch.google_search_links(query, num_results=3)
            if not links or "<error>" in links[0]:
                return "❌ No search results found or failed to fetch results."

            # Fetch content of top 3 links (can be slow; adjust as needed)
            for link in links:
                try:
                    content, _ = await Fetch.fetch_url(link, Fetch.USER_AGENT)
                    fetched_texts.append(f"Content from {link}:\n{content.strip()}")
                except Exception:
                    # Ignore failed fetch for individual links
                    continue
        except Exception as e:
            return f"❌ Search failed: {e}"

    # Compose prompt for synthesis
    context_text = "\n\n---\n\n".join(fetched_texts) if fetched_texts else "No relevant web content found."

    prompt = f"""
You are an AI assistant designed to answer user queries by synthesizing information from multiple trustworthy web sources.

User Query:
\"\"\"{query}\"\"\"

You have gathered the following relevant information from web pages:
\"\"\"{context_text}\"\"\"

Instructions:
- Provide a clear, concise, and well-structured answer to the query.
- Reference important facts from the gathered information.
- Avoid copying large blocks of text; instead, summarize in your own words.
- Mention the URLs or sources when appropriate.
- If the information is insufficient, indicate that clearly.
- Keep the tone helpful, neutral, and informative.

Answer the user query now.
"""
    return await call_ollama(prompt)



async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
