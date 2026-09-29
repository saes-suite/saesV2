"""StatisticalChatAgent for conversational analysis interpretation."""

import os
from typing import Any

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
    ) -> None:
        self.report = report
        self.dataset = dataset
        self.system_prompt = system_prompt
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

        # 1. Check for Anthropic API key
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            response = self._call_anthropic(query, context_str, anthropic_key)
            self.history.append({"role": "assistant", "content": response})
            return response

        # 2. Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            response = self._call_openai(query, context_str, openai_key)
            self.history.append({"role": "assistant", "content": response})
            return response

        # 3. Deterministic expert rule-based interpretation (offline / zero external API dependencies)
        response = self._offline_statistical_interpret(query, context_str)
        self.history.append({"role": "assistant", "content": response})
        return response

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
                        f"because Levene's test detected unequal variances across groups ({p_val_str}, violated homoscedasticity). "
                        "Welch's adjustment adjusts the degrees of freedom via the Welch-Satterthwaite equation, "
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
                    lines.append(
                        f"• **{cell.instance} ({cell.metric})**: {res.effect_size.name} = {res.effect_size.value:.4f} "
                        f"({res.effect_size.magnitude.capitalize()} effect). "
                        f"{'A large effect indicates substantial practical differences beyond pure statistical significance.' if res.effect_size.magnitude == 'large' else 'The differences between algorithms are modest.'}"
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
                    lines.append(f"- {icon} **{a.name}**: statistic={a.statistic:.4f}, p={a.p_value:.4e} ({'Passed' if a.passed else 'Violated'})")
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
                f"- **{cell.instance} ({cell.metric})**: Tested using **{res.test_name}** ($F/t={res.statistic:.3f}$, $p={res.p_value:.4e}$). "
                f"The difference is {sig} at $\\alpha={res.alpha}${eff}."
            )

        summary_lines.append(
            "\n*Tip: Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment to enable full open-ended conversational reasoning.*"
        )
        return "\n".join(summary_lines)

    def _call_anthropic(self, query: str, context: str, api_key: str) -> str:
        """Call Anthropic Claude API for conversational response."""
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=api_key)
            prompt = f"{context}\n\nUser Question: {query}"
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            content = message.content[0]
            return getattr(content, "text", str(content))
        except Exception as e:
            return f"[Claude API unavailable: {e!s}]. Falling back to offline engine:\n\n" + self._offline_statistical_interpret(query, context)

    def _call_openai(self, query: str, context: str, api_key: str) -> str:
        """Call OpenAI API for conversational response."""
        try:
            import openai  # type: ignore

            client = openai.OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{context}\n\nUser Question: {query}"},
                ],
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            return f"[OpenAI API unavailable: {e!s}]. Falling back to offline engine:\n\n" + self._offline_statistical_interpret(query, context)
