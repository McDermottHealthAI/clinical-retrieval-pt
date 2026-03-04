import torch
from meds_torchdata import MEDSTorchBatch

from medrap.batch_adapter import MEDSSupervisedBatch, MEDSSupervisedBatchAdapter
from medrap.encoders import MEDSCodeEncoder


def _example_batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[11, 22, 0], [7, 3, 1]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_meds_code_encoder_uses_batch_code_as_patient_state() -> None:
    encoder = MEDSCodeEncoder()
    batch = _example_batch()

    out = encoder.encode(batch)

    assert torch.equal(out.patient_state, batch.code)


def test_batch_adapter_extracts_boolean_label_from_meds_batch() -> None:
    adapter = MEDSSupervisedBatchAdapter(label_field="boolean_value")
    batch = MEDSTorchBatch(
        code=_example_batch().code,
        numeric_value=_example_batch().numeric_value,
        numeric_value_mask=_example_batch().numeric_value_mask,
        time_delta_days=_example_batch().time_delta_days,
        boolean_value=torch.BoolTensor([True, False]),
    )

    adapted = adapter(batch)

    assert adapted.label_field == "boolean_value"
    assert torch.equal(adapted.targets, torch.BoolTensor([True, False]))
    assert torch.equal(adapted.model_input.code, batch.code)


def test_batch_adapter_extracts_non_boolean_schema_field_from_wrapper() -> None:
    adapter = MEDSSupervisedBatchAdapter(label_field="float_value")
    wrapped = MEDSSupervisedBatch(
        batch=_example_batch(),
        float_value=torch.FloatTensor([0.1, 0.7]),
    )

    adapted = adapter(wrapped)

    assert adapted.label_field == "float_value"
    assert torch.equal(adapted.targets, torch.FloatTensor([0.1, 0.7]))
