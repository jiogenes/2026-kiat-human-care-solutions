# 2026 KIAT Human Care Solutions

각 TC 폴더에서 아래 명령어로 테스트를 실행합니다.

---

## TC-1-2 | 데이터 증강 (호흡 / 압력)

```bash
cd TC-1-2
bash test_case_1_2.sh
```

출력 예시:
```
Respiratory F1 Score : 0.9036
Pressure    F1 Score : 0.9829
Respiratory Accuracy : 0.9158
Pressure    Accuracy : 0.9740
```

---

## TC-3 | Inception Score 측정

```bash
cd TC-3-4
bash test_case_3.sh
```

출력 예시:
```
IS: 87.901966094970703
```

---

## TC-4 | FVD 측정

```bash
cd TC-3-4
bash test_case_4.sh
```

출력 예시:
```
FVD: 32.66455302932304
```

---

## TC-5-6 | 낙상 감지

```bash
cd TC-5-7
bash test_case_5_6.sh
```

출력 예시:
```
Acc: 0.91860465, F1: 0.90909091
```

---

## TC-7 | 낙상 감지 (Active Learning)

```bash
cd TC-5-7
bash test_case_7.sh
```

출력 예시:
```
[AL] Done. Added 116 samples. Active rate: 0.42 Accuracy : 0.9500  F1 score: 0.9524
```
