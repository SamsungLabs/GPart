import multiprocessing
import re
from math import isclose
from typing import Union

from fraction import Fraction
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    # Handle multiple answer patterns (with and without colons)
    answer_patterns = [
        "the answer is: ",
        "the answer is ",
        "answer: ",
        "answer is: ",
        "answer is ",
        "final answer: ",
        "final answer is: ",
        "final answer is ",
    ]

    # Convert to lowercase for pattern matching
    lower_completion = completion.lower()

    # Try each pattern to find the answer
    extract_ans = None
    for pattern in answer_patterns:
        if pattern in lower_completion:
            parts = lower_completion.split(pattern)
            if len(parts) > 1:
                extract_ans = parts[-1].strip()
                break

    # If no pattern found, try to extract the last numeric value
    if extract_ans is None:
        # Look for the last occurrence of common answer indicators
        last_part = lower_completion
        for indicator in ["the answer", "answer", "final answer"]:
            if indicator in lower_completion:
                indicator_parts = lower_completion.split(indicator)
                if len(indicator_parts) > 1:
                    last_part = indicator_parts[-1]
                    break
        extract_ans = last_part.strip()

    if extract_ans:
        # Clean up the extracted answer - remove trailing punctuation and newlines
        extract_ans = extract_ans.split("\n")[0].strip()
        if extract_ans.endswith("."):
            extract_ans = extract_ans[:-1].strip()
        if extract_ans.endswith(","):
            extract_ans = extract_ans[:-1].strip()

        # Extract numeric patterns including fractions, decimals, and integers
        match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract_ans)
        if match:
            matched_number = match.group()
            # Handle fractions
            if "/" in matched_number:
                parts = matched_number.split("/")
                if len(parts) == 2:
                    numerator, denominator = parts
                    if is_number(numerator) and is_number(denominator):
                        if denominator == "0":
                            return None
                        else:
                            try:
                                frac = Fraction(matched_number.replace(",", ""))
                                return round(
                                    float(frac.numerator / frac.denominator), 6
                                )
                            except:
                                return None
            else:
                # Handle regular numbers
                try:
                    cleaned_number = matched_number.replace(",", "")
                    if float(cleaned_number) == float("inf"):
                        return None
                    return round(float(cleaned_number), 6)
                except:
                    return None
        else:
            # If no clean numeric match, try to find any number in the completion
            all_numbers = re.findall(r"[\-+]?\d*[\.,/]?\d+", lower_completion)
            if all_numbers:
                # Return the last number found (most likely to be the answer)
                last_number = all_numbers[-1]
                try:
                    if "/" in last_number:
                        parts = last_number.split("/")
                        if len(parts) == 2:
                            numerator, denominator = parts
                            if (
                                is_number(numerator)
                                and is_number(denominator)
                                and denominator != "0"
                            ):
                                frac = Fraction(last_number.replace(",", ""))
                                return round(
                                    float(frac.numerator / frac.denominator), 6
                                )
                    else:
                        cleaned_number = last_number.replace(",", "")
                        if float(cleaned_number) != float("inf"):
                            return round(float(cleaned_number), 6)
                except:
                    pass

    return None


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    if len(data_list) % batch_size != 0:
        batch_data.append(data_list[n * batch_size :])
    return batch_data


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def is_digit(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(",", ""))
            reference = float(str(reference).replace(",", ""))
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=1e-4):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        (prediction.startswith("[") and prediction.endswith("]"))
        and (reference.startswith("[") and reference.endswith("]"))
        or (prediction.startswith("(") and prediction.endswith(")"))
        and (reference.startswith("(") and reference.endswith(")"))
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()
