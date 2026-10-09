"""Evidence-based AI analytics: report generation and natural-language query routing."""
def answer(question, report, predictions):
    text=question.lower()
    winner=report["winner"]
    if any(word in text for word in ["anomal", "unusual", "outlier"]):
        return f"The isolation forest flags {int(predictions.anomaly.sum())} held-out hours with unusual input conditions. These are review candidates, not confirmed errors."
    if any(word in text for word in ["interval", "uncertain", "coverage"]):
        interval=report["interval"]
        return f"The nominal 90% calibrated interval covers {interval['observed_test_coverage']:.1%} of test observations. Time drift can reduce coverage; the interval is not a guarantee."
    if any(word in text for word in ["peak", "busy", "demand"]):
        row=predictions.loc[predictions.prediction.idxmax()]
        return f"The highest predicted demand in the test period is {row.prediction:.0f} rentals at {row.timestamp}, compared with {row.actual:.0f} observed rentals."
    if any(word in text for word in ["best", "model", "accur", "perform"]):
        return f"{winner} won on validation MAE. Its untouched test MAE is {report['test'][winner]['MAE']:.2f} rentals per hour."
    return "Ask about the best model, peak demand, anomalies, or uncertainty. This local assistant uses measured results and intent matching; it is not a generative language model."
