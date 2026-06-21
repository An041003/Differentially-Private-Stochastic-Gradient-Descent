# DP-SGD GPU tuning notes

- Objective: improve F1 for DP-SGD with `noise_multiplier = 1.5` while tracking epsilon.
- Stop rule: stop when the next tuning round improves best F1 by less than `0.003` or becomes privacy/time inefficient.
- Important: increasing epochs increases epsilon when noise/sample rate are fixed.

## Round notes

- Round 1, max_grad_norm sweep: fixed batch_size=256, epochs=20, lr=0.05, schedule=none; tried clip values 1.0, 1.5, 2.0, 2.5, 3.0. Best in this group was clip=1.0 by F1, with nearby values not improving clearly.
- Round 2, batch_size sweep: kept the best clip from round 1 and tried batch_size 128, 384, 512. batch_size=512 improved F1 substantially compared with 128/384, but epsilon increased to 1.7991.
- Round 3, epoch and schedule sweep: focused on batch_size=512 and tried 30/40 epochs plus cosine/step-like learning-rate behavior. More epochs increased epsilon and did not clearly beat the 20-epoch region.
- Focused extra round around batch_size=512 and lr=0.08 because the first round found F1=0.6661.
- Extra round best F1 improvement was 0.0019. Stopped because the search is now changing F1 by less than 0.003 or requires higher epsilon.
- All max_grad_norm and batch_size changes are recorded in this note and in results/tuning_results.csv.

## Best configuration

- `max_grad_norm`: 2.0
- `batch_size`: 512
- `epochs`: 20
- `learning_rate`: 0.08
- `schedule`: cosine
- epsilon: 1.7991
- accuracy: 0.8531
- F1-score: 0.6680
- ROC-AUC: 0.9072
- PR-AUC: 0.7782
- training time: 19.47s
- device: cuda
- peak GPU memory MB: 133.13

## All tried configurations

- clip=1.0, batch=256, epochs=20, lr=0.05, schedule=none -> status=ok, epsilon=1.21430041194964, accuracy=0.85066401062417, f1=0.652556774293218, time=29.047003746032715
- clip=1.5, batch=256, epochs=20, lr=0.05, schedule=none -> status=ok, epsilon=1.21430041194964, accuracy=0.849800796812749, f1=0.648975791433892, time=28.35740613937378
- clip=2.0, batch=256, epochs=20, lr=0.05, schedule=none -> status=ok, epsilon=1.21430041194964, accuracy=0.849933598937583, f1=0.6510191476219889, time=28.758313179016117
- clip=2.5, batch=256, epochs=20, lr=0.05, schedule=none -> status=ok, epsilon=1.21430041194964, accuracy=0.8494023904382471, f1=0.6518268345102856, time=29.002732038497925
- clip=3.0, batch=256, epochs=20, lr=0.05, schedule=none -> status=ok, epsilon=1.21430041194964, accuracy=0.8470783532536521, f1=0.6503719447396387, time=28.51117491722107
- clip=1.0, batch=128, epochs=20, lr=0.05, schedule=none -> status=ok, epsilon=0.8253931178948875, accuracy=0.845219123505976, f1=0.639554662130818, time=48.67173361778259
- clip=1.0, batch=384, epochs=20, lr=0.05, schedule=none -> status=ok, epsilon=1.523192951315875, accuracy=0.850996015936255, f1=0.65625, time=22.52275800704956
- clip=1.0, batch=512, epochs=20, lr=0.05, schedule=none -> status=ok, epsilon=1.7990588844220654, accuracy=0.851261620185923, f1=0.6656716417910448, time=18.712361335754395
- clip=1.0, batch=512, epochs=30, lr=0.05, schedule=none -> status=ok, epsilon=2.233089124326097, accuracy=0.852257636122178, f1=0.6601496868794868, time=28.222570180892944
- clip=1.0, batch=512, epochs=30, lr=0.05, schedule=cosine -> status=ok, epsilon=2.233089124326097, accuracy=0.851062416998672, f1=0.6619442351168048, time=29.0065393447876
- clip=1.0, batch=512, epochs=40, lr=0.05, schedule=cosine -> status=ok, epsilon=2.608834075818551, accuracy=0.8523240371845949, f1=0.6650602409638554, time=39.091880321502686
- clip=1.0, batch=512, epochs=30, lr=0.03, schedule=cosine -> status=ok, epsilon=2.233089124326097, accuracy=0.848804780876494, f1=0.6557823129251701, time=29.299333095550537
- clip=1.0, batch=512, epochs=30, lr=0.08, schedule=cosine -> status=ok, epsilon=2.233089124326097, accuracy=0.8529880478087649, f1=0.6660633484162896, time=29.83600401878357
- clip=1.0, batch=512, epochs=20, lr=0.08, schedule=none -> status=ok, epsilon=1.7990588844220654, accuracy=0.851593625498008, f1=0.666168782673637, time=19.659510374069214
- clip=1.0, batch=512, epochs=20, lr=0.08, schedule=cosine -> status=ok, epsilon=1.7990588844220654, accuracy=0.8515272244355909, f1=0.6628468033775633, time=19.36511468887329
- clip=1.0, batch=512, epochs=25, lr=0.08, schedule=cosine -> status=ok, epsilon=2.0253857296056412, accuracy=0.851394422310757, f1=0.6629518072289157, time=24.501774787902832
- clip=1.0, batch=512, epochs=25, lr=0.1, schedule=cosine -> status=ok, epsilon=2.0253857296056412, accuracy=0.8533200531208499, f1=0.66706857573474, time=24.159492254257202
- clip=1.0, batch=768, epochs=20, lr=0.05, schedule=none -> status=ok, epsilon=2.2500655743261375, accuracy=0.850199203187251, f1=0.6591115140525838, time=15.79261565208435
- clip=1.0, batch=768, epochs=20, lr=0.08, schedule=cosine -> status=ok, epsilon=2.2500655743261375, accuracy=0.8494023904382471, f1=0.6562594725674447, time=16.14473605155945
- clip=0.5, batch=512, epochs=20, lr=0.08, schedule=cosine -> status=ok, epsilon=1.7990588844220654, accuracy=0.847011952191235, f1=0.6500607533414338, time=18.804343223571777
- clip=2.0, batch=512, epochs=20, lr=0.08, schedule=cosine -> status=ok, epsilon=1.7990588844220654, accuracy=0.8531208499335989, f1=0.6679675773041128, time=19.47454047203064
