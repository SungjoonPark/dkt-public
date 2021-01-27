# Deep Knowledge Tracing

### ckpt
저장된 모델 파라미터를 보관합니다.

### data
assistments : DKT(2015) 논문에 소개된 Assistment2009 데이터입니다. train/test 데이터가 포함되어 있습니다. 각 데이터는 매 3줄이 한 명의 학생을 나타냅니다. 각 학생의 첫번째 줄은 학생이 도전한 문제의 수, 두 번째 줄은 학생이 도전한 문제의 ID, 세 번째 줄은 각각 문제을 풀어서 맞았는지 여부(맞으면 1, 아니면 0)를 나타냅니다.

synthetic : DKT(2015) 논문에 소개된 synthetic-2, syntheic-5 데이터입니다. 총 20개의 버전이 있고, 첫 번째 20개는 문제에서 물어보는 전체 개념의 수가 2(synthetic-2)개, 나머지는 5개(syntheic-5)일 때 입니다. 나머지 5개일 때를 가정하고 시뮬레이션 된 20개의 데이터에 대한 성능 평균이 보고되어 있습니다.


### src
data/ : 학습 데이터를 불러옵니다. 불러온 형태는 loader.py 상단에 있습니다.

loader/ : 볼러온 데이터를 pytorch dataloader로 만듭니다. 만들어진 형태는 dkt.py 상단에 있습니다.

model/ : rnn, lstm 두 가지 모델을 구현합니다. DKT(2015)에 맞게 수정되었습니다.

trainer/ : assistment.py는 Assistment2009 데이터에 대한 학습을 수행하고, simulated.py는 syntheic-5 데이터에 대한 학습을 수행합니다. base.py는 베이스 클래스로 학습에 필요한 공통적인 부분을 구현합니다. metrics.py는 dkt모델의 평가기준인 AUC를 구현합니다.

main.py : 실행에 필요한 argument를 읽고 trainer를 불러와 학습을 시작합니다.


### logs
assist_exp.sh : Assistment2009 데이터에 대한 실험 로그입니다. 재현 성능은 AUC=0.850입니다. 

synth_exp.sh : synthetic-5 데이터에 대한 실험 로그입니다. 재현 성능은 AUC=0.823입니다.

