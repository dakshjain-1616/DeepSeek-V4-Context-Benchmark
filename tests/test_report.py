"""Tests for report module."""

import json

import pytest

from deepseek_v4_context_bench.report import (
    generate_comparison_report,
    generate_csv_report,
    generate_json_report,
    generate_markdown_report,
    generate_report,
)


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report."""

    def test_basic_report(self):
        """Test basic markdown report generation."""
        data = {
            "model": "test-model",
            "corpus_type": "niah",
            "timestamp": "2024-01-01T00:00:00",
            "statistics": {
                "total_tasks": 10,
                "completed_tasks": 8,
                "failed_tasks": 2,
                "accuracy": 0.8,
                "avg_latency_ms": 100.0,
                "total_tokens": 1000,
                "estimated_cost_usd": 0.5,
            },
            "results": [
                {
                    "task_id": "1",
                    "prediction": "answer",
                    "expected_answer": "answer",
                    "score": 1.0,
                    "correct": True,
                    "latency_ms": 100.0,
                    "total_tokens": 100,
                },
            ],
        }
        report = generate_markdown_report(data)
        assert "# DeepSeek V4 Context Benchmark Report" in report
        assert "test-model" in report
        assert "niah" in report
        assert "80.00%" in report or "0.8" in report

    def test_empty_results(self):
        """Test report with empty results."""
        data = {
            "model": "test",
            "corpus_type": "niah",
            "statistics": {
                "total_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "accuracy": 0.0,
                "avg_latency_ms": 0.0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
            "results": [],
        }
        report = generate_markdown_report(data)
        assert "# DeepSeek V4 Context Benchmark Report" in report


class TestGenerateCSVReport:
    """Tests for generate_csv_report."""

    def test_basic_csv(self):
        """Test basic CSV report generation."""
        data = {
            "results": [
                {
                    "task_id": "1",
                    "prediction": "answer",
                    "score": 1.0,
                    "correct": True,
                },
                {
                    "task_id": "2",
                    "prediction": "wrong",
                    "score": 0.0,
                    "correct": False,
                },
            ],
        }
        report = generate_csv_report(data)
        assert "task_id" in report
        assert "1" in report
        assert "2" in report

    def test_empty_results(self):
        """Test CSV with empty results."""
        data = {"results": []}
        report = generate_csv_report(data)
        assert report == ""


class TestGenerateJSONReport:
    """Tests for generate_json_report."""

    def test_pretty_json(self):
        """Test pretty JSON report generation."""
        data = {
            "model": "test",
            "statistics": {"accuracy": 0.8},
        }
        report = generate_json_report(data, pretty=True)
        assert "test" in report
        assert "accuracy" in report
        assert "{\n" in report  # Pretty printed

    def test_compact_json(self):
        """Test compact JSON report generation."""
        data = {"model": "test"}
        report = generate_json_report(data, pretty=False)
        assert "test" in report
        assert "\n" not in report.strip()


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_markdown_format(self, tmp_path):
        """Test generating markdown report from file."""
        data = {
            "model": "test",
            "corpus_type": "niah",
            "statistics": {"total_tasks": 10, "accuracy": 0.8},
            "results": [],
        }
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(data))

        report = generate_report(str(results_file), "markdown")
        assert "# DeepSeek V4 Context Benchmark Report" in report

    def test_json_format(self, tmp_path):
        """Test generating JSON report from file."""
        data = {"model": "test", "statistics": {}, "results": []}
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(data))

        report = generate_report(str(results_file), "json")
        assert "test" in report

    def test_csv_format(self, tmp_path):
        """Test generating CSV report from file."""
        data = {
            "results": [
                {"task_id": "1", "prediction": "answer"},
            ],
        }
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(data))

        report = generate_report(str(results_file), "csv")
        assert "task_id" in report

    def test_invalid_format(self, tmp_path):
        """Test invalid format raises error."""
        data = {"results": []}
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(data))

        with pytest.raises(ValueError):
            generate_report(str(results_file), "invalid")


class TestGenerateComparisonReport:
    """Tests for generate_comparison_report."""

    def test_markdown_comparison(self, tmp_path):
        """Test markdown comparison report."""
        data1 = {
            "model": "model1",
            "corpus_type": "niah",
            "statistics": {"accuracy": 0.8, "avg_latency_ms": 100.0},
        }
        data2 = {
            "model": "model2",
            "corpus_type": "niah",
            "statistics": {"accuracy": 0.9, "avg_latency_ms": 150.0},
        }

        file1 = tmp_path / "results1.json"
        file2 = tmp_path / "results2.json"
        file1.write_text(json.dumps(data1))
        file2.write_text(json.dumps(data2))

        report = generate_comparison_report([str(file1), str(file2)], "markdown")
        assert "# DeepSeek V4 Context Benchmark Comparison" in report
        assert "model1" in report
        assert "model2" in report

    def test_json_comparison(self, tmp_path):
        """Test JSON comparison report."""
        data1 = {"model": "model1", "statistics": {}}
        data2 = {"model": "model2", "statistics": {}}

        file1 = tmp_path / "results1.json"
        file2 = tmp_path / "results2.json"
        file1.write_text(json.dumps(data1))
        file2.write_text(json.dumps(data2))

        report = generate_comparison_report([str(file1), str(file2)], "json")
        parsed = json.loads(report)
        assert len(parsed) == 2
