#!/usr/bin/env python3
"""
Create a comprehensive comparison report between system answers and ground truth.
"""

import json
import re
from pathlib import Path
from datetime import datetime


# Ground truth data
GROUND_TRUTH = {
    1: {
        "question": "What was Apples total revenue for the fiscal year ended September 28, 2024?",
        "answer": "$391,036 million",
        "reference": "Apple 10-K, Item 8, p. 282"
    },
    2: {
        "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?",
        "answer": "15,115,823,000 shares",
        "reference": "Apple 10-K, first paragraph"
    },
    3: {
        "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?",
        "answer": "$96,662 million",
        "reference": "Apple 10-K, Item 8, Note 9, p. 394"
    },
    4: {
        "question": "On what date was Apples 10-K report for 2024 signed and filed with the SEC?",
        "answer": "November 1, 2024",
        "reference": "Apple 10-K, Signature page"
    },
    5: {
        "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?",
        "answer": "No. Checkmark indicates 'No' under Item 1B",
        "reference": "Apple 10-K, Item 1B, p. 176"
    },
    6: {
        "question": "What was Teslas total revenue for the year ended December 31, 2023?",
        "answer": "$96,773 million",
        "reference": "Tesla 10-K, Item 7"
    },
    7: {
        "question": "What percentage of Teslas total revenue in 2023 came from Automotive Sales (excluding Leasing)?",
        "answer": "~84% ($81,924M / $96,773M)",
        "reference": "Tesla 10-K, Item 7"
    },
    8: {
        "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?",
        "answer": "Central to strategy, innovation, leadership; loss could disrupt",
        "reference": "Tesla 10-K, Item 1A"
    },
    9: {
        "question": "What types of vehicles does Tesla currently produce and deliver?",
        "answer": "Model S, Model 3, Model X, Model Y, Cybertruck",
        "reference": "Tesla 10-K, Item 1"
    },
    10: {
        "question": "Tesla's lease pass-through arrangements",
        "answer": "Finance solar systems with investors; customers sign PPAs",
        "reference": "Tesla 10-K, Item 7"
    },
    11: {
        "question": "What is Teslas stock price forecast for 2025?",
        "answer": "Not answerable",
        "reference": "N/A"
    },
    12: {
        "question": "Who is the CFO of Apple as of 2025?",
        "answer": "Not answerable",
        "reference": "N/A"
    },
    13: {
        "question": "What color is Teslas headquarters painted?",
        "answer": "Not answerable",
        "reference": "N/A"
    }
}


def extract_key_answer(answer_text):
    """Extract the key answer from the system's verbose response."""
    # For refused answers
    if "cannot be answered" in answer_text.lower():
        return "REFUSED - Cannot be answered"
    
    # Priority 1: Look for "The final answer is:"
    final_match = re.search(r"The final answer is[:\s]+(.+?)(?:\n|$)", answer_text, re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()[:300]
    
    # Priority 2: Look for "Answer:" at the end
    answer_match = re.search(r"\nAnswer[:\s]+(.+?)(?:\n|$)", answer_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()[:300]
    
    # Priority 3: Look for synthesize section - get first complete sentence
    synth_match = re.search(r"##\s*Synthesize\s*\n(.+?)(?:\n##|\Z)", answer_text, re.IGNORECASE | re.DOTALL)
    if synth_match:
        synth_text = synth_match.group(1).strip()
        # Get first sentence
        sentence_match = re.search(r'^([^.!?]+[.!?])', synth_text)
        if sentence_match:
            return sentence_match.group(1).strip()
        return synth_text.split('\n')[0].strip()[:300]
    
    # Priority 4: Look for specific value patterns in the whole text
    value_patterns = [
        (r"(?:total|is|was|are|were|=)\s+(\$[\d,]+\s*million)", 1),
        (r"(?:is|was|are|were)\s+([\d,]+\s+shares)", 1),
        (r"((?:Model [SXYZ3](?:,\s*)?)+(?:,?\s*(?:and\s+)?(?:Model [SXYZ3]|Cybertruck))?)", 1),
        (r"(November \d+, \d{4})", 1),
        (r"(?:is|was|approximately)\s+(\d+\.?\d*%)", 1),
    ]
    
    for pattern, group_idx in value_patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            return match.group(group_idx).strip()
    
    # Last resort: extract section - first sentence
    extract_match = re.search(r"##\s*Extract\s*\n(.+?)(?:\n##|\Z)", answer_text, re.IGNORECASE | re.DOTALL)
    if extract_match:
        extract_text = extract_match.group(1).strip()
        sentence_match = re.search(r'^([^.!?]+[.!?])', extract_text)
        if sentence_match:
            return sentence_match.group(1).strip()[:300]
    
    # Absolute last resort: first 200 chars
    return ' '.join(answer_text[:200].split())


def compare_answers(system_answer, ground_truth_answer):
    """Compare system answer with ground truth and return status."""
    if ground_truth_answer == "Not answerable":
        if not system_answer or "cannot be answered" in system_answer.lower():
            return "✅ CORRECT (Refused)"
        else:
            return "❌ INCORRECT (Should refuse)"
    
    # Normalize for comparison
    sys_normalized = re.sub(r'[,\s$]', '', system_answer.lower())
    gt_normalized = re.sub(r'[,\s$~]', '', ground_truth_answer.lower())
    
    # Check for key numbers or text
    if gt_normalized in sys_normalized or sys_normalized in gt_normalized:
        return "✅ CORRECT"
    
    # Check if both have same numbers (for revenue, shares, etc.)
    sys_numbers = re.findall(r'\d+', sys_normalized)
    gt_numbers = re.findall(r'\d+', gt_normalized)
    
    if sys_numbers and gt_numbers and any(n in sys_numbers for n in gt_numbers):
        # Close enough
        return "⚠️ PARTIAL"
    
    return "❌ INCORRECT"


def create_html_report(results_file):
    """Create an HTML report with table comparison."""
    
    # Load system results
    with open(results_file, 'r') as f:
        system_results = json.load(f)
    
    # Create comparison data
    comparison = []
    for sys_result in system_results:
        qid = sys_result['question_id']
        gt = GROUND_TRUTH[qid]
        
        system_answer = sys_result['answer']
        key_answer = extract_key_answer(system_answer)
        
        if not sys_result['sources']:
            key_answer = "REFUSED - Cannot be answered"
        
        status = compare_answers(key_answer, gt['answer'])
        
        comparison.append({
            'qid': qid,
            'question': gt['question'],
            'ground_truth': gt['answer'],
            'gt_reference': gt['reference'],
            'system_answer': key_answer,
            'full_system_answer': system_answer,
            'system_sources': ', '.join(sys_result['sources']) if sys_result['sources'] else 'None',
            'status': status
        })
    
    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>RAG System Evaluation Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .summary {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        th {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .correct {
            background-color: #d4edda;
        }
        .incorrect {
            background-color: #f8d7da;
        }
        .partial {
            background-color: #fff3cd;
        }
        .status {
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .expandable {
            cursor: pointer;
            color: #007bff;
            text-decoration: underline;
        }
        .full-answer {
            display: none;
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-left: 3px solid #007bff;
            font-size: 0.9em;
            white-space: pre-wrap;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            font-size: 1.1em;
        }
        .metric-value {
            font-weight: bold;
            font-size: 1.3em;
            color: #4CAF50;
        }
    </style>
    <script>
        function toggleAnswer(id) {
            var elem = document.getElementById('full-' + id);
            if (elem.style.display === 'none') {
                elem.style.display = 'block';
            } else {
                elem.style.display = 'none';
            }
        }
    </script>
</head>
<body>
    <h1>📊 RAG System Evaluation Report</h1>
    <div class="summary">
        <h2>Summary Statistics</h2>
"""
    
    # Calculate metrics
    total = len(comparison)
    correct = sum(1 for c in comparison if '✅' in c['status'])
    incorrect = sum(1 for c in comparison if '❌' in c['status'])
    partial = sum(1 for c in comparison if '⚠️' in c['status'])
    
    answerable = sum(1 for c in comparison if GROUND_TRUTH[c['qid']]['answer'] != 'Not answerable')
    correct_answerable = sum(1 for c in comparison if '✅ CORRECT' == c['status'] and GROUND_TRUTH[c['qid']]['answer'] != 'Not answerable')
    
    html += f"""
        <div class="metric">Total Questions: <span class="metric-value">{total}</span></div>
        <div class="metric">Correct: <span class="metric-value" style="color: #28a745;">{correct}</span></div>
        <div class="metric">Incorrect: <span class="metric-value" style="color: #dc3545;">{incorrect}</span></div>
        <div class="metric">Partial: <span class="metric-value" style="color: #ffc107;">{partial}</span></div>
        <br>
        <div class="metric">Answerable Questions: <span class="metric-value">{answerable}</span></div>
        <div class="metric">Correct Answerable: <span class="metric-value">{correct_answerable}</span></div>
        <div class="metric">Accuracy: <span class="metric-value">{(correct_answerable/answerable*100):.1f}%</span></div>
    </div>
    
    <h2>Detailed Comparison</h2>
    <table>
        <thead>
            <tr>
                <th style="width: 3%;">Q#</th>
                <th style="width: 22%;">Question</th>
                <th style="width: 15%;">Ground Truth</th>
                <th style="width: 15%;">System Answer</th>
                <th style="width: 12%;">Sources</th>
                <th style="width: 8%;">Status</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for item in comparison:
        status_class = ''
        if '✅' in item['status']:
            status_class = 'correct'
        elif '❌' in item['status']:
            status_class = 'incorrect'
        elif '⚠️' in item['status']:
            status_class = 'partial'
        
        html += f"""
            <tr class="{status_class}">
                <td><strong>{item['qid']}</strong></td>
                <td>{item['question']}</td>
                <td><strong>{item['ground_truth']}</strong><br><small style="color: #666;">{item['gt_reference']}</small></td>
                <td>
                    {item['system_answer']}
                    <div class="expandable" onclick="toggleAnswer({item['qid']})">📄 Show full answer</div>
                    <div class="full-answer" id="full-{item['qid']}">{item['full_system_answer']}</div>
                </td>
                <td><small>{item['system_sources']}</small></td>
                <td><span class="status">{item['status']}</span></td>
            </tr>
"""
    
    html += """
        </tbody>
    </table>
    
    <div class="summary" style="margin-top: 30px;">
        <h3>Issues Identified:</h3>
        <ul>
"""
    
    for item in comparison:
        if '❌' in item['status'] or '⚠️' in item['status']:
            html += f"""
            <li><strong>Q{item['qid']}</strong>: {item['question']}<br>
                Expected: {item['ground_truth']}<br>
                Got: {item['system_answer']}</li>
"""
    
    html += """
        </ul>
    </div>
    
    <footer style="margin-top: 40px; padding: 20px; text-align: center; color: #666;">
        <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </footer>
</body>
</html>
"""
    
    return html, comparison


def create_json_report(comparison):
    """Create a structured JSON report."""
    return {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_questions": len(comparison),
            "correct": sum(1 for c in comparison if '✅' in c['status']),
            "incorrect": sum(1 for c in comparison if '❌' in c['status']),
            "partial": sum(1 for c in comparison if '⚠️' in c['status'])
        },
        "detailed_comparison": comparison
    }


if __name__ == "__main__":
    # Find the latest results file
    outputs_dir = Path("outputs")
    result_files = list(outputs_dir.glob("evaluation_results_*.json"))
    
    if not result_files:
        print("No evaluation results found!")
        exit(1)
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Using results from: {latest_file}")
    
    # Create reports
    html_content, comparison_data = create_html_report(latest_file)
    json_report = create_json_report(comparison_data)
    
    # Save HTML report
    html_file = outputs_dir / "evaluation_comparison_report.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    print(f"✅ HTML report saved to: {html_file}")
    
    # Save JSON report
    json_file = outputs_dir / "evaluation_comparison_report.json"
    with open(json_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"✅ JSON report saved to: {json_file}")
    
    print(f"\n📊 Summary:")
    print(f"   Correct: {json_report['summary']['correct']}")
    print(f"   Incorrect: {json_report['summary']['incorrect']}")
    print(f"   Partial: {json_report['summary']['partial']}")
    print(f"\n🌐 Open {html_file} in your browser to view the detailed report!")
