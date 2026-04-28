"""Tests for CLI module."""

import pytest
from click.testing import CliRunner

from deepseek_v4_context_bench.cli import cli, main


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "DeepSeek V4 Context Benchmark" in result.output

    def test_cli_version(self, runner):
        """Test CLI version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_run_help(self, runner):
        """Test run command help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--corpus" in result.output

    def test_report_help(self, runner):
        """Test report command help."""
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output

    def test_card_help(self, runner):
        """Test card command help."""
        result = runner.invoke(cli, ["card", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output

    def test_models_help(self, runner):
        """Test models command help."""
        result = runner.invoke(cli, ["models", "--help"])
        assert result.exit_code == 0

    def test_estimate_help(self, runner):
        """Test estimate command help."""
        result = runner.invoke(cli, ["estimate", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_models_command(self, runner):
        """Test models command."""
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        # Should show model pricing table
        assert "deepseek" in result.output.lower() or "llama" in result.output.lower()

    def test_estimate_command(self, runner):
        """Test estimate command."""
        result = runner.invoke(cli, [
            "estimate",
            "--model", "deepseek/deepseek-v4-flash",
            "--tasks", "10",
            "--tokens", "1000",
        ])
        assert result.exit_code == 0
        assert "Cost" in result.output or "cost" in result.output

    def test_card_command(self, runner, tmp_path):
        """Test card command."""
        output_file = tmp_path / "card.md"
        result = runner.invoke(cli, [
            "card",
            "--output", str(output_file),
            "--title", "Test Card",
        ])
        assert result.exit_code == 0
        assert output_file.exists()

    def test_main_function(self):
        """Test main function entry point."""
        # Just verify it doesn't raise
        import sys
        old_argv = sys.argv
        sys.argv = ["dsv4ctx", "--help"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv

    def test_run_command_dry_run(self, runner):
        """Test run command with dry-run."""
        result = runner.invoke(cli, [
            "run",
            "--model", "deepseek/deepseek-v4-flash",
            "--corpus", "niah",
            "--tasks", "1",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output or "Benchmark complete" in result.output

    def test_run_command_verbose(self, runner):
        """Test run command with verbose flag."""
        result = runner.invoke(cli, [
            "--verbose",
            "run",
            "--model", "deepseek/deepseek-v4-flash",
            "--corpus", "niah",
            "--tasks", "1",
            "--dry-run",
        ])
        assert result.exit_code == 0

    def test_report_command(self, runner, tmp_path):
        """Test report command."""
        # Create a dummy results file
        results_file = tmp_path / "results.json"
        import json
        results_file.write_text(json.dumps({
            "model": "test-model",
            "corpus_type": "niah",
            "total_tasks": 1,
            "completed_tasks": 1,
            "failed_tasks": 0,
            "accuracy": 1.0,
            "avg_latency_ms": 100.0,
            "total_tokens": 100,
            "estimated_cost_usd": 0.1,
        }))
        result = runner.invoke(cli, [
            "report",
            str(results_file),
            "--format", "markdown",
        ])
        assert result.exit_code == 0

    def test_report_command_with_output(self, runner, tmp_path):
        """Test report command with output file."""
        results_file = tmp_path / "results.json"
        import json
        results_file.write_text(json.dumps({
            "model": "test-model",
            "corpus_type": "niah",
            "total_tasks": 1,
            "completed_tasks": 1,
            "failed_tasks": 0,
            "accuracy": 1.0,
            "avg_latency_ms": 100.0,
            "total_tokens": 100,
            "estimated_cost_usd": 0.1,
        }))
        output_file = tmp_path / "report.md"
        result = runner.invoke(cli, [
            "report",
            str(results_file),
            "--format", "markdown",
            "--output", str(output_file),
        ])
        assert result.exit_code == 0
        assert output_file.exists()
