from dgem_nn.networks.dgem_nn_ranker import sum_predictions


def test_average_predictions():
    predictions = {
        "contrast_a": [["drugB", -3.0], ["drugA", -1.7], ["drugD", 1]],
        "contrast_b": [["drugC", -4.0], ["drugA", -1.7], ["drugD", 1]],
        "contrast_c": [["drugE", 5.0], ["drugA", -2.0], ["drugD", 1]],
    }
    aggregated_predictions = sum_predictions(predictions, sort_ascending=True)
    assert len(aggregated_predictions) == 5
    assert aggregated_predictions == [
        ("drugA", -5.4),
        ("drugC", -4.0),
        ("drugB", -3.0),
        ("drugD", 3),
        ("drugE", 5.0),
    ]

    aggregated_predictions = sum_predictions(predictions, sort_ascending=False)
    assert len(aggregated_predictions) == 5
    assert aggregated_predictions == [
        ("drugE", 5.0),
        ("drugD", 3),
        ("drugB", -3.0),
        ("drugC", -4.0),
        ("drugA", -5.4),
    ]
