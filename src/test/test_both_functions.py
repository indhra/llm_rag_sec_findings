#!/usr/bin/env python3
"""
Test script to verify both answer_question functions work correctly:
1. answer_question_validation_comparision - for validation with golden dataset
2. answer_question - for regular Q&A and prompting
"""

from src.pipeline import answer_question, answer_question_validation_comparision

def test_validation_function():
    """Test the validation function with question_id"""
    print("=" * 80)
    print("Testing: answer_question_validation_comparision")
    print("=" * 80)
    
    # Test with question_id (for validation against golden dataset)
    result = answer_question_validation_comparision(
        "What was Apple's total revenue for the fiscal year ended September 28, 2024?",
        question_id=1
    )
    
    print("\nInput:")
    print("  Question: What was Apple's total revenue for the fiscal year ended September 28, 2024?")
    print("  Question ID: 1")
    
    print("\nOutput:")
    print(f"  Question ID: {result.get('question_id')}")
    print(f"  Answer: {result['answer'][:200]}...")
    print(f"  Sources: {result['sources']}")
    
    # Verify structure
    assert 'question_id' in result, "question_id missing from result"
    assert 'answer' in result, "answer missing from result"
    assert 'sources' in result, "sources missing from result"
    assert result['question_id'] == 1, f"Expected question_id=1, got {result['question_id']}"
    
    print("\n✅ Validation function works correctly!")
    return result


def test_regular_qa_function():
    """Test the regular Q&A function without question_id"""
    print("\n" + "=" * 80)
    print("Testing: answer_question")
    print("=" * 80)
    
    # Test without question_id (for regular Q&A)
    result = answer_question(
        "What types of vehicles does Tesla currently produce and deliver?"
    )
    
    print("\nInput:")
    print("  Question: What types of vehicles does Tesla currently produce and deliver?")
    
    print("\nOutput:")
    print(f"  Answer: {result['answer'][:200]}...")
    print(f"  Sources: {result['sources']}")
    
    # Verify structure
    assert 'answer' in result, "answer missing from result"
    assert 'sources' in result, "sources missing from result"
    assert 'question_id' not in result, "question_id should not be in regular Q&A result"
    
    print("\n✅ Regular Q&A function works correctly!")
    return result


def test_out_of_scope():
    """Test both functions with out-of-scope questions"""
    print("\n" + "=" * 80)
    print("Testing: Out-of-Scope Questions")
    print("=" * 80)
    
    # Test validation function with out-of-scope
    result1 = answer_question_validation_comparision(
        "What is Tesla's stock price forecast for 2025?",
        question_id=11
    )
    
    print("\nValidation function (with question_id):")
    print(f"  Question ID: {result1.get('question_id')}")
    print(f"  Answer: {result1['answer']}")
    print(f"  Sources: {result1['sources']}")
    
    assert "cannot be answered" in result1['answer'].lower(), "Should refuse out-of-scope"
    assert result1['sources'] == [], "Sources should be empty for refused questions"
    
    # Test regular Q&A with out-of-scope
    result2 = answer_question(
        "What color is Tesla's headquarters painted?"
    )
    
    print("\nRegular Q&A function (without question_id):")
    print(f"  Answer: {result2['answer']}")
    print(f"  Sources: {result2['sources']}")
    
    assert "cannot be answered" in result2['answer'].lower(), "Should refuse out-of-scope"
    assert result2['sources'] == [], "Sources should be empty for refused questions"
    
    print("\n✅ Both functions correctly refuse out-of-scope questions!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING BOTH ANSWER FUNCTIONS")
    print("=" * 80)
    
    try:
        # Test 1: Validation function
        test_validation_function()
        
        # Test 2: Regular Q&A function
        test_regular_qa_function()
        
        # Test 3: Out-of-scope handling
        test_out_of_scope()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✅")
        print("=" * 80)
        print("\nSummary:")
        print("  ✅ answer_question_validation_comparision works (includes question_id)")
        print("  ✅ answer_question works (no question_id)")
        print("  ✅ Both correctly handle out-of-scope questions")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
