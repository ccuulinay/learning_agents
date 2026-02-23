from IPython.display import display, HTML
from jupyter_server.serverapp import list_running_servers
import os
import json
import logging
import serpapi

# Define helper functions that will be reused throughout the notebook
# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>⚠️ IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) ↗
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix

def serpapi_google_search(query: str) -> dict:
    """Returns the serpapi google search result given query

    Args:
        query (str): The query from agent

    Returns:
        dict: status and result or error msg.
    """
    if query.strip() == "":
        return {
            "status": "error",
            "error_message": (
                f"Sorry, the query is empty, will not perform searching."
            ),
        }
    
    client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))

    try:
        results = client.search({
            'engine': 'google',
            'q': query,
        })
    except serpapi.HTTPError as e:
        _status = "error"
        if e.status_code == 401: # Invalid API key
            logging.exception(e) # "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key"
            _error_message = e.error
        elif e.status_code == 400: # Missing required parameter
            logging.exception(e)
            _error_message = "Missing required parameter"
        elif e.status_code == 429: # Exceeds the hourly throughput limit OR account run out of searches
            logging.exception(e)
            _error_message = "Exceeds the hourly throughput limit OR account run out of searches"
        else:
            logging.exception(e)
            _error_message = str(e)
        return {
             "status": "error",
            "error_message": _error_message
        }
    except serpapi.TimeoutError as e:
        # Handle timeout
        logging.error(f"The request timed out: {e}")
        _error_message = "timeout"
        return {
             "status": "error",
            "error_message": _error_message
        }
    except Exception as e:
        logging.exception(e)
        _error_message = str(e)
        return {
            "status": "error",
            "error_message": _error_message
        }
    if "organic_results" in results:
        _r = json.dumps(results["organic_results"])
    else:
        _r = json.dumps(dict(results))
    
    return {
        "status": "success", "query_result": _r
    }


def execute_python_code(code: str) -> dict:
    """Execute Python code in the local environment and return the result or error."""
    try:
        # A safer way might involve using a dedicated execution environment or sandboxing
        local_scope = {}
        exec(code, globals(), local_scope)
        
        return { "status": "success", "outputs": json.dumps(local_scope) }
    except Exception as e:
        logging.exception(e)
        return { "status": "error", "error_message": str(e) }