def compare_measurements(amateur_measurements, pro_measurements):
    """Compare matching measurements from the amateur and pro videos."""
    comparison_results = {}

    for measurement_name, amateur_value in amateur_measurements.items():
        if measurement_name not in pro_measurements:
            continue

        pro_value = pro_measurements[measurement_name]
        difference = amateur_value - pro_value

        comparison_results[measurement_name] = {
            "amateur": amateur_value,
            "pro": pro_value,
            "difference": difference,
            "absolute_difference": abs(difference),
        }

    return comparison_results


def print_comparison_results(comparison_results):
    """Print measurement comparison results in a beginner-friendly format."""
    if not comparison_results:
        print("\nNo matching measurements were available to compare.")
        return

    print("\nFinal comparison results")
    print("=" * 30)

    for measurement_name, result in comparison_results.items():
        print(f"\nMeasurement: {measurement_name}")
        print(f"Amateur: {result['amateur']:.2f}")
        print(f"Pro: {result['pro']:.2f}")
        print(f"Difference: {result['difference']:.2f}")
        print(f"Absolute difference: {result['absolute_difference']:.2f}")
