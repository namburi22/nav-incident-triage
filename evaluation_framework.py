import json
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import Client
from typing import TypedDict, List, Optional
import time

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
client = Client()

# --- Tools ---
@tool
def get_fund_nav(fund_id: str) -> dict:
    """Get NAV status for a fund."""
    return {
        "FUND001": {"nav": 142.35, "status": "FAILED", "last_updated": "2026-04-14 08:00"},
        "FUND002": {"nav": 98.12, "status": "SUCCESS", "last_updated": "2026-04-14 08:05"},
        "FUND003": {"nav": 0.00, "status": "PENDING", "last_updated": "2026-04-14 07:45"},
    }.get(fund_id, {"error": f"Fund {fund_id} not found"})

@tool
def get_feeds_for_fund(fund_id: str) -> list:
    """Get feed IDs for a fund."""
    return {
        "FUND001": ["FEED_PRICE_01", "FEED_CORP_ACTION"],
        "FUND002": ["FEED_PRICE_02"],
        "FUND003": ["FEED_PRICE_01"],
    }.get(fund_id, [])

@tool
def check_feed_status(feed_id: str) -> dict:
    """Check feed status."""
    return {
        "FEED_PRICE_01": {"status": "DOWN", "down_since": "2026-04-13 22:00", "hours_down": 10},
        "FEED_PRICE_02": {"status": "UP", "last_success": "2026-04-14 08:00"},
        "FEED_CORP_ACTION": {"status": "DELAYED", "delay_mins": 120},
    }.get(feed_id, {"status": "UNKNOWN"})

@tool
def get_impacted_consumers(fund_id: str) -> list:
    """Get downstream consumers."""
    return {
        "FUND001": ["RetailPortal", "AdvisorDashboard", "RegulatoryReporter", "SettlementEngine"],
        "FUND002": ["RetailPortal"],
        "FUND003": ["AdvisorDashboard", "SettlementEngine"],
    }.get(fund_id, [])

@tool
def get_incident_history(fund_id: str) -> list:
    """Get incident history for a fund."""
    return {
        "FUND001": [
            {"date": "2026-03-10", "issue": "Price feed down", "resolution": "Feed restarted", "duration_mins": 45},
            {"date": "2026-02-22", "issue": "Corporate action missing", "resolution": "Manual override", "duration_mins": 120},
        ],
        "FUND003": [
            {"date": "2026-04-01", "issue": "NAV timeout", "resolution": "Reprocessed", "duration_mins": 30}
        ],
    }.get(fund_id, [])

# --- Agent under evaluation ---
async def run_triage_agent(fund_id: str) -> dict:
    """Run the full triage investigation and return structured output."""
    nav = get_fund_nav.invoke({"fund_id": fund_id})
    feed_ids = get_feeds_for_fund.invoke({"fund_id": fund_id})
    feed_statuses = [
        {"feed_id": fid, **check_feed_status.invoke({"feed_id": fid})}
        for fid in feed_ids
    ]
    consumers = get_impacted_consumers.invoke({"fund_id": fund_id})

    async def fund_agent():
        messages = [
            SystemMessage(content="You are a Fund NAV specialist. Be concise and factual."),
            HumanMessage(content=f"""
            Fund: {fund_id}
            NAV Data: {nav}
            Incident History: {get_incident_history.invoke({"fund_id": fund_id})}
            Provide a 3-bullet fund status report.
            """)
        ]
        r = await llm.ainvoke(messages)
        return r.content

    async def feed_agent():
        messages = [
            SystemMessage(content="You are a Data Feed specialist. Be concise and factual."),
            HumanMessage(content=f"""
            Fund: {fund_id}
            Feed Statuses: {feed_statuses}
            Provide a 3-bullet feed health report.
            """)
        ]
        r = await llm.ainvoke(messages)
        return r.content

    async def consumer_agent():
        messages = [
            SystemMessage(content="You are a Downstream Impact specialist. Flag RegulatoryReporter and SettlementEngine as HIGH RISK."),
            HumanMessage(content=f"""
            Fund: {fund_id}
            Impacted Consumers: {consumers}
            Provide a 3-bullet impact report.
            """)
        ]
        r = await llm.ainvoke(messages)
        return r.content

    fund_report, feed_report, consumer_report = await asyncio.gather(
        fund_agent(), feed_agent(), consumer_agent()
    )

    feeds_down = [f for f in feed_statuses if f.get("status") == "DOWN"]
    severity_prompt = f"""
    Classify severity as CRITICAL or STANDARD.
    - Fund: {fund_id}
    - NAV Status: {nav.get('status')}
    - Feeds DOWN: {len(feeds_down)}
    - Consumers: {consumers}

    CRITICAL if: NAV FAILED + feed down 4+ hours + 3+ consumers
    CRITICAL if: RegulatoryReporter or SettlementEngine impacted
    STANDARD: everything else

    Respond in JSON only, no markdown:
    {{"severity": "CRITICAL", "reason": "one sentence"}}
    """
    severity_response = llm.invoke(severity_prompt)
    try:
        content = severity_response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        severity_result = json.loads(content.strip())
    except:
        severity_result = {"severity": "CRITICAL", "reason": "Parse failed"}

    supervisor_messages = [
        SystemMessage(content="You are an Incident Supervisor. Synthesize reports concisely in 3-4 sentences."),
        HumanMessage(content=f"""
        Fund Agent: {fund_report}
        Feed Agent: {feed_report}
        Consumer Agent: {consumer_report}
        What is the situation and how urgent is it?
        """)
    ]
    synthesis = await llm.ainvoke(supervisor_messages)

    return {
        "fund_id": fund_id,
        "nav_status": nav.get("status"),
        "feeds_checked": len(feed_statuses),
        "consumers_impacted": len(consumers),
        "severity": severity_result.get("severity"),
        "severity_reason": severity_result.get("reason"),
        "fund_report": fund_report,
        "feed_report": feed_report,
        "consumer_report": consumer_report,
        "synthesis": synthesis.content,
    }

# --- Test Dataset ---
TEST_CASES = [
    {
        "input": {"fund_id": "FUND001"},
        "expected": {
            "nav_status": "FAILED",
            "severity": "CRITICAL",
            "feeds_checked": 2,
            "consumers_impacted": 4,
            "must_mention": ["FEED_PRICE_01", "SettlementEngine", "RegulatoryReporter"],
            "must_not_mention": ["FUND002", "FUND003"],
        },
        "description": "FUND001 — full critical incident"
    },
    {
        "input": {"fund_id": "FUND002"},
        "expected": {
            "nav_status": "SUCCESS",
            "severity": "STANDARD",
            "feeds_checked": 1,
            "consumers_impacted": 1,
            "must_mention": ["FEED_PRICE_02"],
            "must_not_mention": ["FUND001", "SettlementEngine"],
        },
        "description": "FUND002 — healthy fund, standard severity"
    },
    {
        "input": {"fund_id": "FUND003"},
        "expected": {
            "nav_status": "PENDING",
            "severity": "CRITICAL",
            "feeds_checked": 1,
            "consumers_impacted": 2,
            "must_mention": ["FEED_PRICE_01", "SettlementEngine"],
            "must_not_mention": ["FUND001", "FUND002"],
        },
        "description": "FUND003 — pending NAV, critical due to SettlementEngine"
    },
]

# --- Evaluators ---
def evaluate_severity(result: dict, expected: dict) -> dict:
    actual = result.get("severity", "")
    expected_severity = expected.get("severity", "")
    passed = actual == expected_severity
    return {
        "name": "severity_correct",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": f"Expected {expected_severity}, got {actual}"
    }

def evaluate_nav_status(result: dict, expected: dict) -> dict:
    actual = result.get("nav_status", "")
    expected_status = expected.get("nav_status", "")
    passed = actual == expected_status
    return {
        "name": "nav_status_correct",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": f"Expected {expected_status}, got {actual}"
    }

def evaluate_coverage(result: dict, expected: dict) -> dict:
    full_output = json.dumps(result).lower()
    must_mention = expected.get("must_mention", [])
    missing = [m for m in must_mention if m.lower() not in full_output]
    passed = len(missing) == 0
    return {
        "name": "required_entities_mentioned",
        "passed": passed,
        "score": 1.0 - (len(missing) / max(len(must_mention), 1)),
        "detail": f"Missing: {missing}" if missing else "All required entities mentioned"
    }

def evaluate_no_hallucination(result: dict, expected: dict) -> dict:
    full_output = json.dumps(result).lower()
    must_not = expected.get("must_not_mention", [])
    found = [m for m in must_not if m.lower() in full_output]
    passed = len(found) == 0
    return {
        "name": "no_hallucination",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": f"Found unexpected: {found}" if found else "No hallucinations detected"
    }

def evaluate_feed_count(result: dict, expected: dict) -> dict:
    actual = result.get("feeds_checked", 0)
    expected_count = expected.get("feeds_checked", 0)
    passed = actual == expected_count
    return {
        "name": "feed_count_correct",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": f"Expected {expected_count} feeds, got {actual}"
    }

def evaluate_consumer_count(result: dict, expected: dict) -> dict:
    actual = result.get("consumers_impacted", 0)
    expected_count = expected.get("consumers_impacted", 0)
    passed = actual == expected_count
    return {
        "name": "consumer_count_correct",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": f"Expected {expected_count} consumers, got {actual}"
    }

ALL_EVALUATORS = [
    evaluate_severity,
    evaluate_nav_status,
    evaluate_coverage,
    evaluate_no_hallucination,
    evaluate_feed_count,
    evaluate_consumer_count,
]

# --- LangSmith Logger ---          ← NOW ABOVE run_evaluation
def log_to_langsmith(all_results: list):
    """Log evaluation results to LangSmith for tracking over time."""
    try:
        dataset_name = "nav-triage-eval"
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if datasets:
            dataset = datasets[0]
        else:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="NAV Triage Agent evaluation test cases"
            )
            print(f"[LANGSMITH] Created dataset: {dataset_name}")
            for test in TEST_CASES:
                client.create_example(
                    inputs=test["input"],
                    outputs=test["expected"],
                    dataset_id=dataset.id
                )
            print(f"[LANGSMITH] Uploaded {len(TEST_CASES)} test cases")

        for r in all_results:
            run_id = client.create_run(
                name=f"eval_{r['fund_id']}",
                run_type="chain",
                inputs={"fund_id": r["fund_id"]},
                outputs={
                    "severity": r["agent_output"].get("severity"),
                    "nav_status": r["agent_output"].get("nav_status"),
                    "score": r["score"],
                    "synthesis": r["agent_output"].get("synthesis", "")[:200]
                },
                project_name="nav-incident-triage"
            )
            for e in r["eval_results"]:
                client.create_feedback(
                    run_id=run_id,
                    key=e["name"],
                    score=e["score"],
                    comment=e["detail"]
                )

        print(f"[LANGSMITH] Results logged to project: nav-incident-triage")
        print(f"[LANGSMITH] View at: https://smith.langchain.com")

    except Exception as ex:
        print(f"[LANGSMITH] Logging failed: {ex}")

# --- Run Evaluation ---            ← AFTER log_to_langsmith
async def run_evaluation():
    print("\n" + "="*60)
    print("NAV TRIAGE AGENT — EVALUATION FRAMEWORK")
    print("="*60)
    print(f"Running {len(TEST_CASES)} test cases with {len(ALL_EVALUATORS)} evaluators each")
    print("="*60)

    all_results = []

    for i, test in enumerate(TEST_CASES, 1):
        fund_id = test["input"]["fund_id"]
        expected = test["expected"]
        description = test["description"]

        print(f"\n[TEST {i}/{len(TEST_CASES)}] {description}")
        print(f"  Fund: {fund_id}")

        start = time.time()
        result = await run_triage_agent(fund_id)
        elapsed = time.time() - start

        print(f"  Completed in {elapsed:.2f}s")

        eval_results = []
        for evaluator in ALL_EVALUATORS:
            eval_result = evaluator(result, expected)
            eval_results.append(eval_result)
            status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
            print(f"  {status} {eval_result['name']}: {eval_result['detail']}")

        score = sum(e["score"] for e in eval_results) / len(eval_results)
        all_results.append({
            "test": description,
            "fund_id": fund_id,
            "score": score,
            "passed": sum(1 for e in eval_results if e["passed"]),
            "total": len(eval_results),
            "elapsed": elapsed,
            "eval_results": eval_results,
            "agent_output": result
        })

        print(f"  Score: {score:.0%} ({sum(1 for e in eval_results if e['passed'])}/{len(eval_results)} checks passed)")

    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)

    total_score = sum(r["score"] for r in all_results) / len(all_results)
    total_passed = sum(r["passed"] for r in all_results)
    total_checks = sum(r["total"] for r in all_results)
    avg_time = sum(r["elapsed"] for r in all_results) / len(all_results)

    print(f"\nOverall Score:    {total_score:.0%}")
    print(f"Checks Passed:    {total_passed}/{total_checks}")
    print(f"Avg Agent Time:   {avg_time:.2f}s per fund")
    print(f"\nPer Test Results:")

    for r in all_results:
        status = "✅" if r["score"] == 1.0 else "⚠️" if r["score"] >= 0.8 else "❌"
        print(f"  {status} {r['test']}: {r['score']:.0%} ({r['passed']}/{r['total']})")

    print(f"\n{'='*60}")
    print("REGRESSION CHECK")
    print(f"{'='*60}")
    threshold = 0.85
    if total_score >= threshold:
        print(f"✅ PASSED — Agent quality {total_score:.0%} above threshold {threshold:.0%}")
        print("   Safe to deploy")
    else:
        print(f"❌ FAILED — Agent quality {total_score:.0%} below threshold {threshold:.0%}")
        print("   DO NOT deploy — investigate failing checks")

    failed_checks = []
    for r in all_results:
        for e in r["eval_results"]:
            if not e["passed"]:
                failed_checks.append({
                    "test": r["fund_id"],
                    "check": e["name"],
                    "detail": e["detail"]
                })

    if failed_checks:
        print(f"\nFailed Checks ({len(failed_checks)}):")
        for f in failed_checks:
            print(f"  ❌ [{f['test']}] {f['check']}: {f['detail']}")
    else:
        print("\n🎉 All checks passed — agent is performing correctly")

    print(f"\n{'='*60}")
    print("LOGGING TO LANGSMITH")
    print(f"{'='*60}")
    log_to_langsmith(all_results)

    return all_results

# --- Entry Point ---
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluation())