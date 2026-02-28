import torch
from meds_torchdata import MEDSTorchBatch

from medrap.encoders import MEDSCodeEncoder


def test_meds_code_encoder_uses_batch_code_as_patient_state() -> None:
    encoder = MEDSCodeEncoder()
    batch = MEDSTorchBatch(
        code=torch.LongTensor([[11, 22, 0], [7, 3, 1]]),
        numeric_value=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numeric_value_mask=torch.BoolTensor([[False, False, False], [False, False, False]]),
        time_delta_days=torch.FloatTensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )

    out = encoder.encode(batch)

    assert torch.equal(out.patient_state, batch.code)
