
# import cfd_eval
import cfd_model
# import cloth_eval
import cloth_model
# import deforming_plate_eval
import deforming_plate_model

PARAMETERS = {
  'cfd': dict(
    noise=0.02,
    gamma=1.0,
    field='velocity',
    history=False,
    size=2,
    batch=2,
    model=cfd_model,
    evaluator=None #cfd_eval
  ),
  'cloth': dict(
    noise=0.003,
    gamma=0.1,
    field='world_pos',
    history=True,
    size=3,
    batch=1,
    model=cloth_model,
    evaluator=None #cloth_eval
  ),
  'deforming_plate': dict(
    noise=3e-3,
    gamma=0.1,
    field='world_pos',
    noise_field='world_pos',
    history=False,
    size=3,
    batch=1,
    model=deforming_plate_model,
    evaluator=None #deforming_plate_eval
  ),
}