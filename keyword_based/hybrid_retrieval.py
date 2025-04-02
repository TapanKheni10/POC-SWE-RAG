import json
import os
import re
import subprocess
from typing import Dict, List

import groq
import httpx

# Initialize Groq client
groq_client = groq.Client(
    api_key="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",
    http_client=httpx.Client(verify=False),
)

# Sample data structure to store code snippets
code_snippets = {}  # Map of file_path -> content


def load_codebase(directory: str) -> None:
    """Load code files from a directory into memory."""
    print(f"Loading codebase from {directory}...")

    # Common virtual environment folder patterns to skip
    venv_patterns = ["env", "venv", ".env", ".venv", "virtualenv", "pyenv"]

    # Load all files into memory for keyword search
    for root, dirs, files in os.walk(directory):
        # Skip virtual environment directories
        for venv_dir in venv_patterns:
            if venv_dir in dirs:
                dirs.remove(venv_dir)  # This modifies dirs in-place

        for file in files:
            if file.endswith((".py")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        code_snippets[file_path] = content
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(code_snippets)} files into memory")
    return len(code_snippets)


def extract_keywords(query: str) -> List[str]:
    """Extract keywords from a query using Groq LLM."""
    prompt = f"""
    # Programming Keyword Extraction

    Your task is to extract the key programming terms, function names, variable names, library/module names, and error messages from a programming-related query. Focus on technical terms that would help find relevant code.

    ## Output Format
    Return ONLY a JSON list of keywords. No explanations or additional text.
    Example: ["keyword1", "keyword2", "keyword3"]

    ## Examples

    Query: How do I fix the IndexError in the parse_arguments function when handling empty command-line inputs?
    Keywords: ["IndexError", "parse_arguments", "empty", "command-line", "inputs"]

    ## Your Turn
    
    Query: {query}
    Keywords:
    """

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a code assistant that extracts programming keywords.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        result = response.choices[0].message.content
        # Parse the JSON list of keywords from the result
        try:
            # Find JSON array in the response
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                keywords = json.loads(match.group(0))
                return keywords
            return []
        except:
            # Fall back to simple splitting if JSON parsing fails
            return [word.strip() for word in result.split(",")]
    except Exception as e:
        print(f"Error extracting keywords with Groq: {e}")
        # Fall back to simple keyword extraction
        words = re.findall(r"\b\w+\b", query)
        return [word for word in words if len(word) > 3]


def keyword_based_search(keywords: List[str]) -> List[Dict]:
    results = []

    for keyword in keywords:
        # Search in our in-memory codebase
        for file_path, content in code_snippets.items():
            if keyword.lower() in content.lower():
                # Find the context around the keyword
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if keyword.lower() in line.lower():
                        # Get context (30 lines before and after)
                        start = max(0, i - 30)
                        end = min(len(lines), i + 30)
                        context = "\n".join(lines[start:end])

                        # Calculate match quality score
                        line_words = re.findall(r"\w+", line.lower())
                        score = 0.7  # Base score for containing the keyword

                        if keyword.lower() in line_words:  # Exact word match
                            score = 0.9
                        if re.search(
                            rf"\b{re.escape(keyword.lower())}\b", line.lower()
                        ):  # Word boundary match
                            score = 1.0

                        # Check if it appears to be a function/variable definition
                        if re.search(
                            rf"(def|class|var|let|const|function)\s+{re.escape(keyword)}",
                            line,
                        ):
                            score = 1.0  # Highest score for definition matches

                        results.append(
                            {
                                "file_path": file_path,
                                "content": context,
                                "score": score,
                                "line_number": i + 1,
                                "search_method": "keyword",
                                "matched_keyword": keyword,
                            }
                        )

    # Sort results by score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


def terminal_search(keywords: List[str], directory: str) -> List[Dict]:
    """Use terminal commands (grep/ripgrep) for ultra-fast keyword searches."""
    results = []

    # Common virtual environment patterns to exclude
    venv_exclude_patterns = ["env", "venv", ".env", ".venv", "virtualenv", "pyenv"]

    for keyword in keywords:
        try:
            # Try using ripgrep
            rg_globs = []
            for pattern in venv_exclude_patterns:
                rg_globs.extend(["--glob", f"!**/{pattern}/**"])

            cmd = ["rg", "-i", "-C", "30"] + rg_globs + [keyword, directory]
            completed_process = subprocess.run(cmd, capture_output=True, text=True)

            if completed_process.returncode != 0 and completed_process.returncode != 1:
                # Fall back to grep if ripgrep fails or isn't available
                grep_excludes = []
                for pattern in venv_exclude_patterns:
                    grep_excludes.append(f"--exclude-dir={pattern}")

                cmd = (
                    ["grep", "-r", "-i", "-C", "5"]
                    + grep_excludes
                    + [keyword, directory]
                )
                completed_process = subprocess.run(cmd, capture_output=True, text=True)

            output = completed_process.stdout

            # Process the output into results
            if output:
                file_matches = re.split(r"^\-\-$", output, flags=re.MULTILINE)

                for match in file_matches:
                    match = match.strip()
                    if not match:
                        continue

                    # Parse the file path and content
                    lines = match.split("\n")
                    if lines and ":" in lines[0]:
                        file_path = lines[0].split(":", 1)[0]
                        content = "\n".join(lines)

                        results.append(
                            {
                                "file_path": file_path,
                                "content": content,
                                "score": 1.0,  # Terminal search results are typically high confidence
                                "search_method": "terminal",
                                "matched_keyword": keyword,
                            }
                        )

        except Exception as e:
            print(f"Terminal search error: {e}")
            continue
    return results


def analyze_with_llm(query: str, keyword_results: List[Dict]) -> Dict:
    """Use LLM to analyze keyword search results and suggest code changes."""
    print("Analyzing search results with LLM...")

    # Select the top result for analysis
    if not keyword_results:
        return {"error": "No code snippets found to analyze"}

    top_result = keyword_results[0]
    file_path = top_result["file_path"]

    # Get a wider context window (100 lines) around the match
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()

        line_number = top_result["line_number"]
        start_line = max(0, line_number - 50)
        end_line = min(len(all_lines), line_number + 50)
        wider_context = "".join(all_lines[start_line:end_line])

        # Create line number mapping for the snippet
        actual_line_numbers = list(range(start_line + 1, end_line + 1))
    except Exception as e:
        print(f"Error getting wider context: {e}")
        wider_context = top_result["content"]
        actual_line_numbers = []

    # Prepare prompt for LLM
    prompt = f"""
    # Code Analysis and Fix Suggestion
    
    ## Issue Description
    {query}
    
    ## File Path
    {file_path}
    
    ## Code Snippet
    ```python
    {wider_context}
    ```
    
    ## Task
    Analyze this code snippet and suggest specific changes to fix the issue described.
    
    Your response should include:
    1. The exact line numbers that need to be modified
    2. The current code at those lines
    3. The suggested replacement code
    4. A brief explanation of why this change fixes the issue
    
    Note: Ensure that all whitespace, including indentation levels, is preserved exactly as it appears in the original code.
    
    Format your response as a JSON object with the following structure:
    {{
        "changes": [
            {{
                "line_number": 123,
                "original_code": "current code at this line",
                "suggested_code": "fixed code for this line",
                "explanation": "Brief explanation of why this change is needed"
            }}
        ],
        "summary": "Overall explanation of the fix"
    }}
    """

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a code analysis assistant that identifies bugs and suggests fixes.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        result = response.choices[0].message.content

        # Extract JSON from response
        try:
            match = re.search(r"({.*})", result, re.DOTALL)
            if match:
                analysis = json.loads(match.group(0))

                # Adjust line numbers if we have the mapping
                if actual_line_numbers:
                    for change in analysis.get("changes", []):
                        relative_line = change.get("line_number", 0)
                        if 0 <= relative_line - 1 < len(actual_line_numbers):
                            change["line_number"] = actual_line_numbers[
                                relative_line - 1
                            ]

                return analysis
            return {
                "error": "Could not parse LLM response as JSON",
                "raw_response": result,
            }
        except Exception as e:
            return {"error": f"Error parsing LLM response: {e}", "raw_response": result}

    except Exception as e:
        return {"error": f"Error calling LLM API: {e}"}


def apply_suggested_changes(analysis: Dict, file_path: str) -> Dict:
    """Apply the suggested changes to the actual file."""
    if "error" in analysis:
        return {"success": False, "message": f"Error in analysis: {analysis['error']}"}

    if "changes" not in analysis or not analysis["changes"]:
        return {"success": False, "message": "No changes to apply"}

    try:
        # Read the entire file
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Track which lines we've modified
        modified_lines = []

        # Apply each change
        for change in analysis["changes"]:
            line_number = change.get("line_number")
            original_code = change.get("original_code")
            suggested_code = change.get("suggested_code")

            if not all([line_number, original_code, suggested_code]):
                continue

            # Convert to 0-based index
            line_idx = line_number - 1

            if 0 <= line_idx < len(lines):
                # Check if the original code matches (at least partially)
                # if original_code.strip() in lines[line_idx].strip():
                # Replace the line
                lines[line_idx] = suggested_code + "\n"
                modified_lines.append(line_number)

        if not modified_lines:
            return {
                "success": False,
                "message": "No lines were modified - original code not found",
            }

        # Write the modified content back to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return {
            "success": True,
            "message": f"Successfully applied {len(modified_lines)} changes to {file_path}",
            "modified_lines": modified_lines,
        }

    except Exception as e:
        return {"success": False, "message": f"Error applying changes: {str(e)}"}


def find_relevant_files(keywords: List[str], directory: str) -> List[Dict]:
    """Use terminal commands to find relevant files containing keywords."""
    file_matches = {}  # Dictionary to track files and their matched keywords

    # Common virtual environment patterns to exclude
    venv_exclude_patterns = ["env", "venv", ".env", ".venv", "virtualenv", "pyenv"]

    for keyword in keywords:
        try:
            # Try using ripgrep with file names only
            rg_globs = []
            for pattern in venv_exclude_patterns:
                rg_globs.extend(["--glob", f"!**/{pattern}/**"])

            # Use -l flag for listing files only
            cmd = ["rg", "-i", "-l"] + rg_globs + [keyword, directory]
            completed_process = subprocess.run(cmd, capture_output=True, text=True)

            if completed_process.returncode != 0 and completed_process.returncode != 1:
                # Fall back to grep if ripgrep fails or isn't available
                grep_excludes = []
                for pattern in venv_exclude_patterns:
                    grep_excludes.append(f"--exclude-dir={pattern}")

                cmd = ["grep", "-r", "-i", "-l"] + grep_excludes + [keyword, directory]
                completed_process = subprocess.run(cmd, capture_output=True, text=True)

            output = completed_process.stdout

            # Process the output into results - just file paths
            if output:
                file_paths = output.strip().split("\n")

                for file_path in file_paths:
                    if file_path.strip():
                        if file_path not in file_matches:
                            file_matches[file_path] = []
                        file_matches[file_path].append(keyword)

        except Exception as e:
            print(f"Terminal search error: {e}")
            continue

    # Convert to list format with metadata
    results = []
    for file_path, matched_keywords in file_matches.items():
        results.append(
            {
                "file_path": file_path,
                "matched_keywords": matched_keywords,
                "keyword_count": len(matched_keywords),
                "score": len(matched_keywords),  # Initial score based on keyword count
            }
        )

    # Sort by keyword count (most matched keywords first)
    results.sort(key=lambda x: x["keyword_count"], reverse=True)

    return results


def rank_files_with_llm(query: str, file_candidates: List[Dict]) -> List[Dict]:
    """Use LLM to rank file candidates based on relevance to the query."""
    if not file_candidates:
        return []

    # Sort candidates by initial keyword match count for fallback ordering
    sorted_candidates = sorted(
        file_candidates, key=lambda x: x["keyword_count"], reverse=True
    )

    print("Ranking files with LLM...: ")

    # Prepare file info for the LLM - using all files
    file_info = []
    id_to_index_map = {}  # To map LLM ids back to indices

    for i, file in enumerate(sorted_candidates):
        file_info.append(
            {
                "id": i + 1,
                "file_path": file["file_path"],
                "matched_keywords": file["matched_keywords"],
            }
        )
        id_to_index_map[i + 1] = i

    prompt = f"""
    # File Ranking for Code Search
    
    ## Query
    {query}
    
    ## Available Files (found by keyword search)
    {json.dumps(file_info, indent=2)}
    
    ## Task
    Rank these files by their likely relevance to solving the query.
    
    Return a JSON array of file IDs in order of relevance (most relevant first).
    For example: [3, 1, 5, 2, 4] if file ID 3 is most likely to be relevant.
    
    You don't need to include all files in your response - just focus on the top 10-20 most relevant ones.
    
    Provide ONLY the JSON array, no additional text.
    """

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a code search assistant that helps find the most relevant files for solving programming issues.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        result = response.choices[0].message.content

        # Parse the JSON list from the result
        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                ranked_ids = json.loads(match.group(0))

                # Reorder based on LLM ranking
                ranked_files = []
                for id in ranked_ids:
                    if id in id_to_index_map:
                        ranked_files.append(sorted_candidates[id_to_index_map[id]])

                # Add any remaining files that weren't ranked
                ranked_ids_set = set(ranked_ids)
                for i in range(len(sorted_candidates)):
                    if i + 1 not in ranked_ids_set:
                        ranked_files.append(sorted_candidates[i])

                return ranked_files

        except Exception as e:
            print(f"Error parsing LLM ranking response: {e}")

    except Exception as e:
        print(f"Error getting LLM ranking: {e}")

    # Fall back to original order if LLM ranking fails
    return sorted_candidates


def extract_code_snippets(file_path: str, keywords: List[str]) -> List[Dict]:
    """Extract code snippets around keywords in a file with 100-line window using ripgrep."""
    snippets = []

    try:
        for keyword in keywords:
            try:
                # Use ripgrep to find matches with context
                cmd = ["rg", "-i", "-C", "50", "--json", keyword, file_path]
                completed_process = subprocess.run(cmd, capture_output=True, text=True)

                # If ripgrep succeeded or had no matches (code 1)
                if completed_process.returncode in [0, 1]:
                    output = completed_process.stdout

                    # Process JSON output
                    for line in output.strip().split("\n"):
                        if not line:
                            continue

                        try:
                            result = json.loads(line)

                            # We're only interested in match results
                            if result.get("type") == "match":
                                match_data = result.get("data", {})
                                line_number = match_data.get("line_number", 0)

                                # Get the context using another ripgrep call to avoid parsing issues
                                context_cmd = [
                                    "rg",
                                    "-i",
                                    "-C",
                                    "50",
                                    keyword,
                                    file_path,
                                ]
                                context_process = subprocess.run(
                                    context_cmd, capture_output=True, text=True
                                )

                                if context_process.returncode in [0, 1]:
                                    context = context_process.stdout

                                    # Approximate window start/end
                                    window_start = max(1, line_number - 50)
                                    window_end = line_number + 50

                                    snippets.append(
                                        {
                                            "file_path": file_path,
                                            "content": context,
                                            "line_number": line_number,
                                            "matched_keyword": keyword,
                                            "score": 1.0,  # Base score
                                            "window_start": window_start,
                                            "window_end": window_end,
                                        }
                                    )
                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                print(f"Ripgrep search error for {keyword} in {file_path}: {e}")

                # Fall back to the original implementation if ripgrep fails
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                keyword_lower = keyword.lower()
                for i, line in enumerate(lines):
                    if keyword_lower in line.lower():
                        start_line = max(0, i - 50)
                        end_line = min(len(lines), i + 50)
                        context = "".join(lines[start_line:end_line])

                        snippets.append(
                            {
                                "file_path": file_path,
                                "content": context,
                                "line_number": i + 1,
                                "matched_keyword": keyword,
                                "score": 1.0,  # Base score
                                "window_start": start_line + 1,
                                "window_end": end_line,
                            }
                        )

    except Exception as e:
        print(f"Error extracting code snippets from {file_path}: {e}")

    print("-" * 80)
    print(f"Extracted {len(snippets)} snippets from {file_path}")

    return snippets


def hybrid_retrieval(
    query: str, codebase_dir: str, auto_apply_changes: bool = False
) -> List[Dict]:
    print("Starting search process...")

    # 1. Extract keywords using Groq
    keywords = extract_keywords(query)
    print(f"Extracted keywords: {keywords}")

    if not keywords:
        print("No keywords extracted, cannot proceed with search")
        return []

    # 2. Find files that contain the keywords
    print("Finding relevant files...")
    relevant_files = find_relevant_files(keywords, codebase_dir)
    print(f"Found {len(relevant_files)} files containing keywords")

    if not relevant_files:
        print("No relevant files found for the given query")
        return []

    # 3. Use LLM to rank files by relevance
    ranked_files = rank_files_with_llm(query, relevant_files)
    print(f"Ranked {len(ranked_files)} files by relevance")

    # 4. Extract code snippets from top-ranked files
    all_snippets = []
    max_files_to_process = min(5, len(ranked_files))  # Process top 5 files

    print(f"Extracting code snippets from top {max_files_to_process} files...")
    for file_data in ranked_files[:max_files_to_process]:
        file_path = file_data["file_path"]
        print(f"Processing file: {file_path}")

        file_snippets = extract_code_snippets(file_path, file_data["matched_keywords"])

        if file_snippets:
            all_snippets.extend(file_snippets)
            print(f"Found {len(file_snippets)} code snippets in {file_path}")

    # 5. Sort snippets by score
    all_snippets.sort(key=lambda x: x["score"], reverse=True)

    print(
        f"Found {len(all_snippets)} code snippets across {max_files_to_process} files"
    )

    # If we found snippets, analyze with LLM
    if all_snippets:
        print("✅ Found relevant code snippets. Analyzing...")

        # Add LLM analysis of the results to suggest fixes
        analysis = analyze_with_llm(query, all_snippets)

        if "error" not in analysis:
            print("*" * 100)
            print("LLM analysis complete. Suggested changes:")
            print(json.dumps(analysis, indent=2))

            print("*" * 100)

            # Add the analysis to the first result
            if all_snippets:
                all_snippets[0]["suggested_changes"] = analysis

                # Auto-apply changes if requested
                if auto_apply_changes:
                    file_path = all_snippets[0]["file_path"]
                    print(f"Attempting to apply changes to {file_path}...")
                    result = apply_suggested_changes(analysis, file_path)
                    print(result["message"])

                    if result["success"]:
                        all_snippets[0]["changes_applied"] = True
                        all_snippets[0]["modified_lines"] = result["modified_lines"]
        else:
            print(f"LLM analysis error: {analysis['error']}")

        return all_snippets[:10]  # Return top 10 snippets
    else:
        print("No relevant code snippets found for the given query")
        return []


def main():
    # Example codebase directory
    codebase_dir = "/Users/harshabajaj/Desktop/SWE-bench/test"
    # codebase_dir = "/Users/harshabajaj/Desktop/SWE-bench/new_test_folder"

    # Sample SWE Bench issue/query
    swe_bench_query = """
    Solve the error SyntaxError in the main function. I am not able to run the file.
    """

    # Load code from the directory
    files = load_codebase(codebase_dir)
    if files > 0:
        results = hybrid_retrieval(
            swe_bench_query, codebase_dir, auto_apply_changes=True
        )

        # Display results
        print(f"\nResults for query: {swe_bench_query}")
        print("-" * 80)

        for i, result in enumerate(results[:5]):  # Show top 5 results
            print(f"File: {result['file_path']}")
            if result.get("type") and result.get("name"):
                print(f"Type: {result['type']}, Name: {result['name']}")

            # Display suggested changes if available
            if result.get("suggested_changes"):
                print("\nSuggested Changes:")
                changes = result["suggested_changes"]

                if "summary" in changes:
                    print(f"Summary: {changes['summary']}")

                for change in changes.get("changes", []):
                    print(f"\nLine {change['line_number']}:")
                    print(f"Original: {change['original_code']}")
                    print(f"Suggested: {change['suggested_code']}")
                    print(f"Explanation: {change['explanation']}")

                # Show if changes were applied
                if result.get("changes_applied"):
                    print(
                        f"\n✅ Changes successfully applied to lines: {', '.join(map(str, result['modified_lines']))}"
                    )

            print("-" * 80)
    else:
        print("No files loaded")


if __name__ == "__main__":
    main()
