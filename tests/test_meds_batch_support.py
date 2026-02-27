import torch
from meds_torchdata import MEDSTorchBatch

from clinical_retrieval_pt.encoders import MEDSCodeEncoder


def test_meds_code_encoder_uses_batch_code_as_patient_state() -> None:
    encoder = MEDSCodeEncoder()
    batch = MEDSTorchBatch(
        code=torch.LongTensor([[11, 22, 0], [7, 3, 1]]),
        numeric_value=torch.zeros((2, 3), dtype=torch.float32),
        numeric_value_mask=torch.zeros((2, 3), dtype=torch.bool),
        time_delta_days=torch.zeros((2, 3), dtype=torch.float32),
    )

    out = encoder.encode(batch)

    assert torch.equal(out.patient_state, batch.code)
