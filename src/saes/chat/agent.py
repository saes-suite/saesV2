import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import httpx

from saes.app.pipeline import AnalysisReport
from saes.chat.context import build_analysis_context
from saes.chat.prompts import STATISTICAL_SYSTEM_PROMPT
from saes.domain.models import Dataset


class StatisticalChatAgent:
    """Conversational assistant for interpreting SAES statistical evaluation results."""

    def __init__(
        self,
        report: AnalysisReport | None = None,
        dataset: Dataset | None = None,
        system_prompt: str = STATISTICAL_SYSTEM_PROMPT,
        provider: str = "offline",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.report = report
        self.dataset = dataset
        self.system_prompt = system_prompt
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.history: list[dict[str, str]] = []

    def set_context(self, report: AnalysisReport | None, dataset: Dataset | None = None) -> None:
        """Update current statistical context."""
        self.report = report
        if dataset is not None:
            self.dataset = dataset

    def ask(self, query: str) -> str:
        """Process user query against the current statistical analysis context.

        Args:
            query: Question or prompt from user.

        Returns:
            String response explaining the results or answering the question.
        """
        self.history.append({"role": "user", "content": query})
        context_str = build_analysis_context(self.report, self.dataset)

        provider = (self.provider or "offline").lower().strip()

        # 1. Local AI model (Ollama / LocalAI / LM Studio)
        if provider == "local":
            response = self._call_local(
                query=query,
                context=context_str,
                model=self.model,
                base_url=self.base_url,
            )
            self.history.append({"role": "assistant", "content": response})
            return response

        # 2. OpenAI provider
        if provider == "openai":
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            if key:
                response = self._call_openai(
                    query=query,
                    context=context_str,
                    api_key=key,
                    model=self.model,
                    base_url=self.base_url,
                )
            else:
                fallback = self._offline_statistical_interpret(query, context_str)
                response = f"[OpenAI API key missing]. Falling back to offline statistical engine:\n\n{fallback}"
            self.history.append({"role": "assistant", "content": response})
            return response

        # 3. Anthropic Claude provider
        if provider == "anthropic":
            key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if key:
                response = self._call_anthropic(
                    query=query,
                    context=context_str,
                    api_key=key,
                    model=self.model,
                )
            else:
                fallback = self._offline_statistical_interpret(query, context_str)
                response = f"[Anthropic API key missing]. Falling back to offline statistical engine:\n\n{fallback}"
            self.history.append({"role": "assistant", "content": response})
            return response

        # 4. Google Gemini provider
        if provider in ("gemini", "google"):
            key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if key:
                response = self._call_gemini(
                    query=query,
                    context=context_str,
                    api_key=key,
                    model=self.model,
                )
            else:
                fallback = self._offline_statistical_interpret(query, context_str)
                response = f"[Google Gemini API key missing]. Falling back to offline statistical engine:\n\n{fallback}"
            self.history.append({"role": "assistant", "content": response})
            return response

        # 5. Explicit Offline mode
        if provider == "offline":
            response = self._offline_statistical_interpret(query, context_str)
            self.history.append({"role": "assistant", "content": response})
            return response

        # 5. Fallback auto-detection for backwards compatibility
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            response = self._call_anthropic(query, context_str, anthropic_key)
            self.history.append({"role": "assistant", "content": response})
            return response

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            response = self._call_openai(query, context_str, openai_key)
            self.history.append({"role": "assistant", "content": response})
            return response

        response = self._offline_statistical_interpret(query, context_str)
        self.history.append({"role": "assistant", "content": response})
        return response

    def stream_ask(self, query: str) -> Iterator[str]:
        """Process user query and yield response tokens iteratively.

        Args:
            query: Question or prompt from user.

        Yields:
            Token or text chunk strings as they are generated.
        """
        self.history.append({"role": "user", "content": query})
        context_str = build_analysis_context(self.report, self.dataset)
        provider = (self.provider or "offline").lower().strip()

        chunks: list[str] = []

        def _gather(gen: Iterator[str]) -> Iterator[str]:
            for chunk in gen:
                chunks.append(chunk)
                yield chunk

        if provider == "local":
            gen = self._stream_local(query, context_str, model=self.model, base_url=self.base_url)
        elif provider == "openai":
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            if key:
                gen = self._stream_openai(query, context_str, api_key=key, model=self.model, base_url=self.base_url)
            else:
                gen = self._stream_missing_key("OpenAI", query, context_str)
        elif provider == "anthropic":
            key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if key:
                gen = self._stream_anthropic(query, context_str, api_key=key, model=self.model)
            else:
                gen = self._stream_missing_key("Anthropic", query, context_str)
        elif provider in ("gemini", "google"):
            key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if key:
                gen = self._stream_gemini(query, context_str, api_key=key, model=self.model)
            else:
                gen = self._stream_missing_key("Google Gemini", query, context_str)
        elif provider == "offline":
            gen = self._stream_offline(query, context_str)
        else:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            if anthropic_key:
                gen = self._stream_anthropic(query, context_str, api_key=anthropic_key, model=self.model)
            elif openai_key:
                gen = self._stream_openai(query, context_str, api_key=openai_key, model=self.model)
            else:
                gen = self._stream_offline(query, context_str)

        try:
            yield from _gather(gen)
        finally:
            full_response = "".join(chunks)
            if full_response:
                self.history.append({"role": "assistant", "content": full_response})

    def _stream_missing_key(self, provider_name: str, query: str, context: str) -> Iterator[str]:
        prefix = f"[{provider_name} API key missing]. Falling back to offline statistical engine:\n\n"
        yield prefix
        yield from self._stream_offline(query, context)

    def _stream_offline(self, query: str, context: str) -> Iterator[str]:
        full_text = self._offline_statistical_interpret(query, context)
        tokens = re.split(r"(\s+)", full_text)
        is_test = bool(os.getenv("PYTEST_CURRENT_TEST"))
        for token in tokens:
            if not token:
                continue
            yield token
            if not is_test:
                time.sleep(0.003)

    def _stream_local(
        self,
        query: str,
        context: str,
        model: str | None = None,
        base_url: str | None = None,
    ) -> Iterator[str]:
        if not model or not model.strip():
            yield (
                "[No model specified for Local AI. Please connect to Ollama and select a model in Settings]. "
                "Falling back to offline statistical engine:\n\n"
            )
            yield from self._stream_offline(query, context)
            return

        selected_model = model.strip()
        clean_url = (base_url or "http://localhost:11434").rstrip("/")
        prompt = f"{context}\n\nUser Question: {query}"

        try:
            with httpx.Client(timeout=60.0) as client:
                if clean_url.endswith("/v1"):
                    v1_payload: dict[str, Any] = {
                        "model": selected_model,
                        "stream": True,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    }
                    with client.stream("POST", f"{clean_url}/chat/completions", json=v1_payload) as res:
                        res.raise_for_status()
                        for line in res.iter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {}).get("content", "")
                                    if delta:
                                        yield delta
                            except Exception:
                                continue
                else:
                    ollama_payload: dict[str, Any] = {
                        "model": selected_model,
                        "stream": True,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    }
                    with client.stream("POST", f"{clean_url}/api/chat", json=ollama_payload) as res:
                        res.raise_for_status()
                        for line in res.iter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                msg = chunk.get("message", {})
                                content = msg.get("content", "")
                                if content:
                                    yield content
                                if chunk.get("done", False):
                                    break
                            except Exception:
                                continue
        except Exception as e:
            yield (
                f"[Local AI error on {clean_url} (model: {selected_model}): {e!s}]. "
                "Ensure Ollama or your local server is running.\n\n"
                "Falling back to offline statistical engine:\n\n"
            )
            yield from self._stream_offline(query, context)

    def _stream_openai(
        self,
        query: str,
        context: str,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
    ) -> Iterator[str]:
        if not model or not model.strip():
            yield (
                "[No model specified for OpenAI. Please select a model in Settings]. "
                "Falling back to offline statistical engine:\n\n"
            )
            yield from self._stream_offline(query, context)
            return

        selected_model = model.strip()
        prompt = f"{context}\n\nUser Question: {query}"

        try:
            try:
                import openai

                client = openai.OpenAI(api_key=api_key, base_url=base_url or None)
                stream = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except ImportError:
                pass

            target_url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": selected_model,
                "stream": True,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", target_url, headers=headers, json=payload) as res:
                    res.raise_for_status()
                    for line in res.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                        except Exception:
                            continue
        except Exception as e:
            yield f"[OpenAI API unavailable: {e!s}]. Falling back to offline engine:\n\n"
            yield from self._stream_offline(query, context)

    def _stream_anthropic(
        self,
        query: str,
        context: str,
        api_key: str,
        model: str | None = None,
    ) -> Iterator[str]:
        if not model or not model.strip():
            yield (
                "[No model specified for Anthropic Claude. Please select a model in Settings]. "
                "Falling back to offline statistical engine:\n\n"
            )
            yield from self._stream_offline(query, context)
            return

        selected_model = model.strip()
        prompt = f"{context}\n\nUser Question: {query}"

        try:
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
                with client.messages.stream(
                    model=selected_model,
                    max_tokens=1000,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        yield text
                return
            except ImportError:
                pass

            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": selected_model,
                "max_tokens": 1000,
                "stream": True,
                "system": self.system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST", "https://api.anthropic.com/v1/messages", headers=headers, json=payload
                ) as res:
                    res.raise_for_status()
                    for line in res.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        try:
                            chunk = json.loads(data_str)
                            if chunk.get("type") == "content_block_delta":
                                text = chunk.get("delta", {}).get("text", "")
                                if text:
                                    yield text
                        except Exception:
                            continue
        except Exception as e:
            yield f"[Claude API unavailable: {e!s}]. Falling back to offline engine:\n\n"
            yield from self._stream_offline(query, context)

    def _stream_gemini(
        self,
        query: str,
        context: str,
        api_key: str,
        model: str | None = None,
    ) -> Iterator[str]:
        if not model or not model.strip():
            yield (
                "[No model specified for Google Gemini. Please connect and select an available model in Settings]. "
                "Falling back to offline statistical engine:\n\n"
            )
            yield from self._stream_offline(query, context)
            return

        selected_model = model.strip().replace("models/", "")
        prompt = f"{context}\n\nUser Question: {query}"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{selected_model}:streamGenerateContent?key={api_key}&alt=sse"
        )
        payload: dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": self.system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", url, json=payload) as res:
                    res.raise_for_status()
                    for line in res.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        try:
                            chunk = json.loads(data_str)
                            candidates = chunk.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for part in parts:
                                    text = part.get("text", "")
                                    if text:
                                        yield text
                        except Exception:
                            continue
        except Exception as e:
            yield f"[Google Gemini API unavailable: {e!s}]. Falling back to offline engine:\n\n"
            yield from self._stream_offline(query, context)

    def _offline_statistical_interpret(self, query: str, context: str) -> str:
        """Intelligent offline statistical interpretation engine when no cloud LLM key is configured."""
        q_lower = query.lower()

        if self.report is None or not self.report.cell_analyses:
            return (
                "No statistical analysis has been executed yet. "
                "Please run an analysis on your dataset first, then ask questions about the tests, "
                "assumptions, or effect sizes."
            )

        # Query: Why Welch / test selection
        if "why" in q_lower or "welch" in q_lower or "test selection" in q_lower or "choose" in q_lower:
            explanations: list[str] = []
            for cell in self.report.cell_analyses:
                res = cell.result
                if "Welch" in res.test_name:
                    levene_res = next((a for a in res.assumptions if "Levene" in a.name), None)
                    p_val_str = f"p={levene_res.p_value:.3e}" if levene_res else "p < alpha"
                    explanations.append(
                        f"• For **{cell.instance} ({cell.metric})**, **{res.test_name}** was chosen "
                        f"because Levene's test detected unequal variances across groups ({p_val_str}). "
                        "Welch's correction adjusts degrees of freedom via the Welch-Satterthwaite equation, "
                        "preventing inflated Type I error rates."
                    )
                else:
                    explanations.append(
                        f"• For **{cell.instance} ({cell.metric})**, **{res.test_name}** was used because "
                        "the normality and equal variance assumptions were satisfied (or groups were balanced)."
                    )
            return "### Test Selection Rationale\n\n" + "\n\n".join(explanations)

        # Query: Effect size / practical significance
        if "effect size" in q_lower or "practical" in q_lower or "magnitude" in q_lower:
            lines: list[str] = []
            for cell in self.report.cell_analyses:
                res = cell.result
                if res.effect_size:
                    mag_desc = (
                        "A large effect indicates substantial practical differences."
                        if res.effect_size.magnitude == "large"
                        else "The differences between algorithms are modest."
                    )
                    lines.append(
                        f"• **{cell.instance} ({cell.metric})**: {res.effect_size.name} = {res.effect_size.value:.4f} "
                        f"({res.effect_size.magnitude.capitalize()} effect). {mag_desc}"
                    )
                else:
                    lines.append(f"• **{cell.instance} ({cell.metric})**: No effect size computed.")
            return "### Effect Size & Practical Significance Interpretation\n\n" + "\n".join(lines)

        # Query: Assumptions (normality / homoscedasticity)
        if "assumption" in q_lower or "normal" in q_lower or "variance" in q_lower or "levene" in q_lower:
            lines = []
            for cell in self.report.cell_analyses:
                res = cell.result
                lines.append(f"#### `{cell.instance}` — `{cell.metric}`")
                for a in res.assumptions:
                    icon = "✅" if a.passed else "⚠️"
                    status = "Passed" if a.passed else "Violated"
                    lines.append(f"- {icon} **{a.name}**: statistic={a.statistic:.4f}, p={a.p_value:.4e} ({status})")
            return "### Statistical Assumptions Evaluation\n\n" + "\n".join(lines)

        # Default summary interpretation
        summary_lines = [
            "### Statistical Findings Summary",
            f"Evaluated {len(self.report.cell_analyses)} experimental conditions (cells):",
        ]
        for cell in self.report.cell_analyses:
            res = cell.result
            sig = "statistically significant" if res.is_significant else "not statistically significant"
            eff = f" with {res.effect_size.magnitude} effect ({res.effect_size.value:.3f})" if res.effect_size else ""
            summary_lines.append(
                f"- **{cell.instance} ({cell.metric})**: Tested using **{res.test_name}** "
                f"($F/t={res.statistic:.3f}$, $p={res.p_value:.4e}$). "
                f"The difference is {sig} at $\\alpha={res.alpha}${eff}."
            )

        summary_lines.append(
            "\n*Tip: Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to enable open-ended reasoning.*"
        )
        return "\n".join(summary_lines)

    @staticmethod
    def fetch_local_models(base_url: str = "http://localhost:11434") -> list[str]:
        """Fetch available models from a local Ollama server."""
        clean_url = (base_url or "http://localhost:11434").rstrip("/")
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{clean_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            pass
        return []

    def _call_local(
        self,
        query: str,
        context: str,
        model: str | None = None,
        base_url: str | None = None,
    ) -> str:
        """Call local LLM (Ollama or OpenAI-compatible server)."""
        if not model or not model.strip():
            fallback = self._offline_statistical_interpret(query, context)
            return (
                "[No model specified for Local AI. Please connect to Ollama and select a model in Settings]. "
                f"Falling back to offline statistical engine:\n\n{fallback}"
            )
        selected_model = model.strip()
        clean_url = (base_url or "http://localhost:11434").rstrip("/")
        prompt = f"{context}\n\nUser Question: {query}"

        try:
            with httpx.Client(timeout=60.0) as client:
                if clean_url.endswith("/v1"):
                    v1_payload: dict[str, Any] = {
                        "model": selected_model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    }
                    res = client.post(f"{clean_url}/chat/completions", json=v1_payload)
                    res.raise_for_status()
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return str(choices[0].get("message", {}).get("content", ""))
                    return ""
                else:
                    ollama_payload: dict[str, Any] = {
                        "model": selected_model,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    }
                    res = client.post(f"{clean_url}/api/chat", json=ollama_payload)
                    res.raise_for_status()
                    data = res.json()
                    return str(data.get("message", {}).get("content", ""))
        except Exception as e:
            fallback = self._offline_statistical_interpret(query, context)
            return (
                f"[Local AI error on {clean_url} (model: {selected_model}): {e!s}]. "
                f"Ensure Ollama or your local server is running.\n\n"
                f"Falling back to offline statistical engine:\n\n{fallback}"
            )

    def _call_anthropic(
        self,
        query: str,
        context: str,
        api_key: str,
        model: str | None = None,
    ) -> str:
        """Call Anthropic Claude API for conversational response."""
        if not model or not model.strip():
            fallback = self._offline_statistical_interpret(query, context)
            return (
                "[No model specified for Anthropic Claude. Please select a model in Settings]. "
                f"Falling back to offline statistical engine:\n\n{fallback}"
            )
        selected_model = model.strip()
        prompt = f"{context}\n\nUser Question: {query}"
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=selected_model,
                max_tokens=1000,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            content = message.content[0]
            return str(getattr(content, "text", str(content)))
        except ImportError:
            try:
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload: dict[str, Any] = {
                    "model": selected_model,
                    "max_tokens": 1000,
                    "system": self.system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                }
                with httpx.Client(timeout=45.0) as client:
                    res = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                    res.raise_for_status()
                    data = res.json()
                    content_list = data.get("content", [])
                    if content_list:
                        return str(content_list[0].get("text", ""))
                    return ""
            except Exception as e:
                fallback = self._offline_statistical_interpret(query, context)
                return f"[Claude API unavailable: {e!s}]. Falling back to offline engine:\n\n{fallback}"
        except Exception as e:
            fallback = self._offline_statistical_interpret(query, context)
            return f"[Claude API unavailable: {e!s}]. Falling back to offline engine:\n\n{fallback}"

    def _call_openai(
        self,
        query: str,
        context: str,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
    ) -> str:
        """Call OpenAI API or OpenAI-compatible custom endpoint."""
        if not model or not model.strip():
            fallback = self._offline_statistical_interpret(query, context)
            return (
                "[No model specified for OpenAI. Please select a model in Settings]. "
                f"Falling back to offline statistical engine:\n\n{fallback}"
            )
        selected_model = model.strip()
        prompt = f"{context}\n\nUser Question: {query}"
        try:
            import openai

            client = openai.OpenAI(api_key=api_key, base_url=base_url or None)
            completion = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return str(completion.choices[0].message.content or "")
        except ImportError:
            try:
                target_url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload: dict[str, Any] = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                }
                with httpx.Client(timeout=45.0) as client:
                    res = client.post(target_url, headers=headers, json=payload)
                    res.raise_for_status()
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return str(choices[0].get("message", {}).get("content", ""))
                    return ""
            except Exception as e:
                fallback = self._offline_statistical_interpret(query, context)
                return f"[OpenAI API unavailable: {e!s}]. Falling back to offline engine:\n\n{fallback}"
        except Exception as e:
            fallback = self._offline_statistical_interpret(query, context)
            return f"[OpenAI API unavailable: {e!s}]. Falling back to offline engine:\n\n{fallback}"

    def _call_gemini(
        self,
        query: str,
        context: str,
        api_key: str,
        model: str | None = None,
    ) -> str:
        """Call Google Gemini REST API."""
        if not model or not model.strip():
            fallback = self._offline_statistical_interpret(query, context)
            return (
                "[No model specified for Google Gemini. Please connect and select an available model in Settings]. "
                f"Falling back to offline statistical engine:\n\n{fallback}"
            )
        selected_model = model.strip().replace("models/", "")
        prompt = f"{context}\n\nUser Question: {query}"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{selected_model}:generateContent?key={api_key}"
        )
        payload: dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": self.system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
        }
        try:
            with httpx.Client(timeout=45.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return str(parts[0].get("text", ""))
                return ""
        except Exception as e:
            fallback = self._offline_statistical_interpret(query, context)
            return f"[Google Gemini API unavailable: {e!s}]. Falling back to offline engine:\n\n{fallback}"

    @staticmethod
    def start_ollama_service() -> bool:
        """Attempt to start the local Ollama background daemon."""
        binary_path = shutil.which("ollama")
        if not binary_path:
            for p in ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama", "/usr/bin/ollama"]:
                if os.path.exists(p):
                    binary_path = p
                    break
        if binary_path:
            try:
                subprocess.Popen(
                    [binary_path, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(1.5)
                return True
            except Exception:
                return False
        return False

    @staticmethod
    def probe_provider(
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Probe AI provider connectivity and retrieve available models."""
        prov = (provider or "offline").lower().strip()

        if prov == "offline":
            return {
                "status": "ok",
                "models": ["deterministic-statistical-engine"],
                "message": "Deterministic statistical inference engine active (Zero cloud API dependencies).",
                "can_start_ollama": False,
            }

        if prov == "local":
            clean_url = (base_url or "http://localhost:11434").rstrip("/")
            models = StatisticalChatAgent.fetch_local_models(clean_url)
            if models:
                return {
                    "status": "ok",
                    "models": models,
                    "message": f"Connected to Ollama ({len(models)} model(s) available).",
                    "can_start_ollama": False,
                }
            # Check if running but has 0 models
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"{clean_url}/api/tags")
                    if resp.status_code == 200:
                        return {
                            "status": "ok",
                            "models": [],
                            "message": "Connected to Ollama, but no models installed. Run: ollama run llama3.2",
                            "can_start_ollama": False,
                        }
            except Exception:
                pass

            # Not running, check if binary is installed
            binary = shutil.which("ollama")
            if not binary:
                for p in ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama", "/usr/bin/ollama"]:
                    if os.path.exists(p):
                        binary = p
                        break
            if binary:
                return {
                    "status": "binary_found_not_running",
                    "models": [],
                    "message": f"Ollama is installed at {binary}, but the service is currently stopped.",
                    "can_start_ollama": True,
                    "binary_path": binary,
                }
            return {
                "status": "not_installed",
                "models": [],
                "message": "Ollama is not installed or not in PATH.",
                "can_start_ollama": False,
            }

        if prov in ("gemini", "google"):
            key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
            if not key:
                return {
                    "status": "error",
                    "models": [],
                    "message": "Google Gemini API key is missing. Please enter your API key.",
                }
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
                with httpx.Client(timeout=10.0) as client:
                    res = client.get(url)
                    if res.status_code in (400, 403):
                        return {
                            "status": "error",
                            "models": [],
                            "message": "Invalid Google Gemini API key or unauthorized project.",
                        }
                    res.raise_for_status()
                    data = res.json()
                    models = [
                        m["name"].replace("models/", "")
                        for m in data.get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    preferred = [
                        "gemini-3.5-flash",
                        "gemini-3.5-flash-lite",
                        "gemini-3.8-flash",
                        "gemini-flash-latest",
                        "gemini-2.5-pro",
                    ]
                    sorted_models = [m for m in preferred if m in models] + [m for m in models if m not in preferred]
                    return {
                        "status": "ok",
                        "models": sorted_models,
                        "message": f"Successfully connected to Google Gemini ({len(models)} models available).",
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "models": [],
                    "message": f"Failed to connect to Google Gemini: {e!s}",
                }

        if prov == "openai":
            key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
            if not key:
                return {
                    "status": "error",
                    "models": [],
                    "message": "OpenAI API key is missing.",
                }
            try:
                clean_url = (base_url or "https://api.openai.com/v1").rstrip("/")
                target_url = f"{clean_url}/models"
                headers = {"Authorization": f"Bearer {key}"}
                with httpx.Client(timeout=10.0) as client:
                    res = client.get(target_url, headers=headers)
                    if res.status_code == 401:
                        return {
                            "status": "error",
                            "models": [],
                            "message": "Invalid OpenAI API key.",
                        }
                    res.raise_for_status()
                    data = res.json()
                    chat_models = [
                        m["id"]
                        for m in data.get("data", [])
                        if any(k in m["id"] for k in ["gpt-4", "gpt-3.5", "o1", "o3", "chat"])
                    ]
                    preferred = ["gpt-4o-mini", "gpt-4o", "o3-mini", "gpt-4.5-preview"]
                    sorted_models = [m for m in preferred if m in chat_models] + [
                        m for m in sorted(chat_models) if m not in preferred
                    ]
                    return {
                        "status": "ok",
                        "models": sorted_models,
                        "message": f"Successfully connected to OpenAI ({len(sorted_models)} models available).",
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "models": [],
                    "message": f"Failed to connect to OpenAI: {e!s}",
                }

        if prov == "anthropic":
            key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
            if not key:
                return {
                    "status": "error",
                    "models": [],
                    "message": "Anthropic API key is missing.",
                }
            try:
                headers = {
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                }
                with httpx.Client(timeout=10.0) as client:
                    res = client.get("https://api.anthropic.com/v1/models", headers=headers)
                    if res.status_code == 401:
                        return {
                            "status": "error",
                            "models": [],
                            "message": "Invalid Anthropic API key.",
                        }
                    if res.status_code == 200:
                        data = res.json()
                        models = [m["id"] for m in data.get("data", [])]
                        return {
                            "status": "ok",
                            "models": models,
                            "message": f"Successfully connected to Anthropic Claude ({len(models)} models available).",
                        }
                    return {
                        "status": "error",
                        "models": [],
                        "message": f"Failed to fetch models from Anthropic (HTTP {res.status_code}).",
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "models": [],
                    "message": f"Failed to connect to Anthropic: {e!s}",
                }

        return {
            "status": "error",
            "models": [],
            "message": f"Unknown provider: {provider}",
        }

    @staticmethod
    def pull_ollama_model(name: str, base_url: str = "http://localhost:11434") -> dict[str, str]:
        """Request the local Ollama daemon to download/pull a model."""
        clean_url = (base_url or "http://localhost:11434").rstrip("/")
        clean_name = name.strip()
        if not clean_name:
            return {"status": "error", "message": "Model name cannot be empty."}
        try:
            with httpx.Client(timeout=600.0) as client:
                res = client.post(f"{clean_url}/api/pull", json={"name": clean_name, "stream": False})
                if res.status_code == 200:
                    return {
                        "status": "ok",
                        "message": f"Model '{clean_name}' downloaded successfully into Ollama.",
                    }
                return {
                    "status": "error",
                    "message": f"Ollama pull returned HTTP {res.status_code}: {res.text}",
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to pull model '{clean_name}': {e!s}",
            }

    @staticmethod
    def stream_pull_ollama_model(
        name: str,
        base_url: str = "http://localhost:11434",
    ) -> Iterator[str]:
        """Stream progress chunks of Ollama model pull."""
        clean_url = (base_url or "http://localhost:11434").rstrip("/")
        clean_name = name.strip()
        if not clean_name:
            yield json.dumps({"status": "error", "error": "Model name cannot be empty."}) + "\n"
            return
        try:
            with httpx.Client(timeout=1800.0) as client:
                with client.stream(
                    "POST", f"{clean_url}/api/pull", json={"name": clean_name, "stream": True}
                ) as response:
                    if response.status_code != 200:
                        yield (
                            json.dumps(
                                {"status": "error", "error": f"Ollama returned HTTP {response.status_code}"}
                            )
                            + "\n"
                        )
                        return
                    for line in response.iter_lines():
                        if line:
                            yield f"{line}\n"
        except Exception as e:
            yield json.dumps({"status": "error", "error": str(e)}) + "\n"

