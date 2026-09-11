"""Small SDK server adapter. Business logic and schemas stay in agent_tools."""
import inspect
import json
from functools import partial

import anyio
from jsonschema import Draft202012Validator, FormatChecker
from mcp import types
from mcp.server.lowlevel import Server

from app import agent_tools
from app.mcp.context import current_context


def build_server(name, tool_names):
    server = Server(name)
    schemas = {n: agent_tools.public_schema(n) for n in tool_names}

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=n, description=s["description"], inputSchema=s["parameters"])
                for n, s in schemas.items()]

    # Validate ourselves to avoid the SDK returning argument values in errors.
    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        try:
            if name not in schemas:
                raise ValueError("Unknown tool")
            args = dict(arguments or {})
            Draft202012Validator(schemas[name]["parameters"], format_checker=FormatChecker()).validate(args)
            context = current_context.get()
            if name in {"create_task", "log_mood"} and (
                context is None or name not in context.allowed_writes
            ):
                raise PermissionError("User request does not authorize this write")
            fn = agent_tools.TOOL_FUNCTIONS[name]
            params = inspect.signature(fn).parameters
            if "session_name" in params:
                if context is None or not context.session_name:
                    raise PermissionError("Session required")
                args["session_name"] = context.session_name
            if "document_id" in params:
                if context is None or not context.document_id:
                    raise PermissionError("Document required")
                args["document_id"] = context.document_id
            # anyio propagates the active trace and trusted context to this worker.
            result = await anyio.to_thread.run_sync(partial(fn, **args))
            if isinstance(result, dict) and (result.get("error") or result.get("success") is False or result.get("logged") is False):
                raise RuntimeError("Service unavailable")
            structured = result if isinstance(result, dict) else {"results": result}
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(result, default=str))],
                structuredContent=json.loads(json.dumps(structured, default=str)),
            )
        except Exception as exc:
            failure = {"error": "Tool unavailable or invalid request", "error_type": type(exc).__name__}
            return types.CallToolResult(isError=True, structuredContent=failure,
                content=[types.TextContent(type="text", text=json.dumps(failure))])

    return server
