# CentrixSupport MCP setup

CentrixSupport uses the official Python MCP SDK. The default `builtin`
transport creates an MCP client session over the SDK's in-memory transport for
each server. Tool discovery uses `tools/list`, and execution uses `tools/call`.
The tool handlers delegate to the existing services in `app/agent_tools.py`.

Install dependencies with:

```powershell
pip install -r requirements.txt
```

The default configuration needs no extra process:

```dotenv
MCP_DATA_SERVER=builtin
MCP_KNOWLEDGE_SERVER=builtin
MCP_WEB_SERVER=builtin
MCP_SERVER_TIMEOUT_SECONDS=20
```

Each server variable may instead contain the URL of a compatible MCP
Streamable HTTP endpoint. The agent consumes discovered schemas, so adding or
changing tools on an external server does not require edits to
`app/agent_runner.py`. External data servers remain responsible for enforcing
the same identity and authorization policy as the built-in server.

Set `MCP_HEALTH_TOKEN` to enable the internal `GET /health/mcp` endpoint. Call
it with `Authorization: Bearer <token>`. Its response contains connection and
discovery status, counts, timestamps, and latency; it does not return endpoint
URLs, arguments, credentials, or prompts.

The web server exposes only `search_web(query, num_results=5)`. It sends the
query to DuckDuckGo's fixed HTML search endpoint and never fetches result URLs.
