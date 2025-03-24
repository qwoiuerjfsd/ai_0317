
//메뉴 생성기 종료 E==============================
//데이터 아키텍처{sub_title:"",sub_content:"",sub_img:[],user_fill:""}
let data_sets=[]
class DataSet{
	constructor(sub_title,menuNum){this.sub_title=sub_title}
	user_fill=""
	sub_content=[]
	sub_img=[]
	set_content(content){this.sub_content.push(content)}
	set_img(num,obj){
		if(!this.sub_img[num]){this.sub_img[num]=[]}
		this.sub_img[num].push(obj)
	}
	set_fill(ufill){this.user_fill=ufill}	 
}


let d1 = new DataSet("선형회귀모델")//메인 타이틀 //메뉴번호
d1.set_content("보스턴 주택 가격 예측 선형 모델")//서브 타이틀
d1.set_img(0,{imgtitle:"보스턴 데이터 수신",imgurl:"https://drive.google.com/file/d/1TgZwp61x45vVZhOPSeN9-IIDLfPRmCbx/view?usp=drive_link",imglog:"텐서플로우 보스턴 데이터셋 수신 코드",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"데이터 특성 파악",imglog:"각 필드별 데이터의 특성의 값을 확인",imgurl:"https://drive.google.com/file/d/1rDEH6nJee7Hi4dEqyKcX8ZWACJZvYIMP/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"데이터 연관성 확인",imglog:"가격정답과 데이터의 특성별 상호 연관도를 파악",imgurl:"https://drive.google.com/file/d/13ltn5tQw0cPusa9DrLR9VtX9bIIERg4P/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"데이터 분포도 파악",imglog:"히스토그램을 이용하여 데이터의 분포와 이상치 데이터 확인",imgurl:"https://drive.google.com/file/d/1nrjXsYPJ97cWZmi7OPwR2FZZDimSXBAp/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"데이터 정규분포 전환",imglog:"훈련 전처리를 위한 연관성이 있는 데이터를 평균 0 표준편차 1로 구성된 정규분포로 변환 ",imgurl:"https://drive.google.com/file/d/1Eg-BMsE2MD8EPVyG1U6E6_R3WvYTkMA9/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"순차모델구성 및 훈련실행",imglog:"평균제곱오차법을 이용한 손실함수와 경사하강법을 이용한 최적화 함수로 컴파일 및 최적화된 훈련 15회 실행",imgurl:"https://drive.google.com/file/d/1e9l-PzORpHwkW-uW-B5bBCtXTJe4was4/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"훈련결과 시각화",imglog:"훈련 결과인 mse 손실값의 시각화 표현",imgurl:"https://drive.google.com/file/d/1t6Nm-ZYC0WZf6Rnz4xZCAffDl12Rhs1B/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
d1.set_img(0,{imgtitle:"모델의 예측결과 측정",imglog:"테스트 데이터를 주입하여 예측결과를 인출하고 실제 정답과 차이를 정확률로 표시",imgurl:"https://drive.google.com/file/d/1YdeZKYxbEB4IDbzX3_F7u4_mJibbMYvR/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀




d1.set_content("캘리포니아 주택 가격 예측 선형 회귀모델")
d1.set_img(1,{imgtitle:"캘리포니아 주택 가격 예측 선형 회귀모델",imgurl:"",imglog:"",sourceurl:""})
d1.set_img(1,{imgtitle:"1년후 당뇨상태 예측",imgurl:"",imglog:"",sourceurl:'https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py'})//이미지타이틀
	d1.set_img(1,{imgtitle:"캘리포니아 주택 특성데이터 수신 및 분석",imglog:"사이킷런에서 제공하는 캐리포니아 주택 가격에 따른 데이터 특성(X)들의 모음과 그에 따른 가격정보(Y) ",imgurl:"https://drive.google.com/file/d/1RDiWwhAZ3GnuM5iEXbLw7xlm64ukgxjS/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀
	d1.set_img(1,{imgtitle:"주택 특성과 가격의 연관성 분석",imglog:"주택의 특설병 산전도 분석으로 선형성 확인",imgurl:"https://drive.google.com/file/d/1zA2NIHgD9dJ6WTENLpLbzP44NLp03bTe/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})
	d1.set_img(1,{imgtitle:"데이터 통계정보 분석",imglog:"판다스 데이터프레임으로 전화후 평균치, 표준편차등의 데이터 통계정보 분석",imgurl:"https://drive.google.com/file/d/1Ct42X89y40ml4R1WoCb0f_XCDa0S7hAL/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})
	d1.set_img(1,{imgtitle:"데이터 분포 확인",imglog:"히스토그램으로 데이터 분포 시각화와 이상데이터 또는 범위를 벗어난 데이터 설정",imgurl:"https://drive.google.com/file/d/1j0XBzliGyvphrzqJ-ZkTs3V-Ayw5yfLY/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})
	d1.set_img(1,{imgtitle:"이상데이터제거",imglog:"범위를 벗어나거나 이상치 데이터는 성능에 치명적인 영향을 줄 수 있으므로 이상치 및 범위를 벗어난 데이터를 제거하여 데이터 정체를 수행",imgurl:"https://drive.google.com/file/d/1kb3QOK339MbXsSTIrPT3g794DwBRxAjC/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})
	d1.set_img(1,{imgtitle:"데이터정제후 분포확인",imglog:"데이터 전처리 시행후 데이터 범위등의 이상데이터 히스토그램으로 분포확인",imgurl:"https://drive.google.com/file/d/1tLRYbENLnFQ-rkfgPY7EL7yx53KaGL3q/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})
	d1.set_img(1,{imgtitle:"훈련데이터와 테스트데이터 분할",imglog:"훈련데이터 80% 데스트 데이터20%를 사이킷런 라이브러리를 이용하여 분할 및 데이터 정규분포화 실행",imgurl:"https://drive.google.com/file/d/1r6Z2cASZHgkiVeSm7WBTAlTlJiXS5BOd/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})
	d1.set_img(1,{imgtitle:"선형회귀 기계학습 모델 구성과 훈련",imglog:"은닉층이 존재하지 않는 머신러닝 모델을 구성하고 평균절대오차 손실함수 설정과 경사하강법으로 최적화 함수를 설정한후 훈련 100회 실행",imgurl:"https://drive.google.com/file/d/1jf8dfpCrMWgvuamEH8vsj7Kzx-D5zldL/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})
	d1.set_img(1,{imgtitle:"훈련결과 시각화",imglog:"훈련시 저장된 손실값을 이용하여 시각화 그래프 표현",imgurl:"https://drive.google.com/file/d/1kwBrKsKOLa3tX8hg-eAsYiSN6kViWqQc/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})
	


d1.set_content("")

d1.set_fill("선형 회귀 모델은 단일데이터를 이용하거나 다중 데이터를 이용하여 연속적인 값을 출력하여 예측한다")//사용자 에필로그
data_sets.push(d1)

// menu2 =============================================================
let d2 = new DataSet("분류모델구현")//메인타이틀
d2.set_content("패션 mnist 회귀 다중 분류")//서브 타이틀
d2.set_img(0,{imgtitle:"fashion_mnist 데이터 수신",imglog:"1.구글에서 제공하는 드레스,셔츠,샌달등의 패션관련 이미지 다운로드",imgurl:"https://drive.google.com/file/d/1phdpGsOpth89fs_I8JMhap9ir9N4zA83/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"수신데이터 확인",imglog:"2.훈련데이터 6만개, 테스트 데이터 1만개, 이미지 사이즈 28,28 1채널 그레이, 정답데이터 정수형",imgurl:"https://drive.google.com/file/d/1wD1xZr4c9kohQWoQn4psJSyQSMxDZKpl/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"훈련데이터 minmax 정규환",imglog:"3.훈련데이터의 0 ~ 1 사이값으로 정규화 실행",imgurl:"https://drive.google.com/file/d/1e6iRHf_bijkDExYSaaM5gSFyUBL2-71P/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"데이터셔플링 및 정답일치성확인",imglog:"4.데이터셔플링 후 정답과 일치하도록 셔플이 되었는지 확인 작업",imgurl:"https://drive.google.com/file/d/1QyYbntpxdXsUjq0gBTAYAcVd58GHeTkL/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"커스텀원핫인코딩 수행",imglog:"5.커스텀 원핫인코딩 클래스 생성후 원하는 방법으로 원핫이코딩 후 작동 확인",imgurl:"https://drive.google.com/file/d/1Z3AUtAk3_wCAdXgZo5Rb01MMlxWqr8pN/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"모델컴파일및 훈련실행",imglog:"6.flatten으로 완전층 연결후 다중분류모델로 softmax 활성화 함수로 10class 출력",imgurl:"https://drive.google.com/file/d/149vCUICt_Ope7HJUyaEZ3X5cTgJkJXsX/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"훈련결과 시각화",imglog:"7.훈련 종류후 손실 및 정확도 시각화",imgurl:"https://drive.google.com/file/d/1EPvWnm6_9J0Wi04OntXroA2Rni4am-uG/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"테스트 데이터 예측",imglog:"8.모델의 테스트데이터 예측과 예측결과 시각화",imgurl:"https://drive.google.com/file/d/1Y8_YrKd_J6SVJQ5NzIJI3jDpZr3TDraP/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"모델 및 인코더 저장",imglog:"9.훈련된 모델의 저장과 레이블이 연결된 인코더 파일로 저장",imgurl:"https://drive.google.com/file/d/1bCAPPmxienEDJc5HJQhm3yw9RtjALiBd/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"모델과 인코더 불러오기",imglog:"실체 이미지 측정을 위해 모델과 훈련된 모델과 레이블 라벨 인코더 다시 불러오기",imgurl:"https://drive.google.com/file/d/1BopnRe5Ziv2ACnYWdN6iWW7EwKDqjn3t/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"실제이미지 테스트",imglog:"인터넷의 이미지를 복사하여(test_ing) 모델의 적합하도록 전처리후 예측값 출력",imgurl:"https://drive.google.com/file/d/139p7uBuYoXArR7Opsa7Z0Tr6JqgDSNhb/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"실제 이미지 예측결과 시각화",imglog:"실제 이미지를 예측한 결과를 시각화 하여 표현",imgurl:"https://drive.google.com/file/d/1DsBrTqKrwb5MXSFa-W8luCkyyzFwjFPh/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀
d2.set_img(0,{imgtitle:"",imglog:"",imgurl:"",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/classificationSoftmax_fashionmlist/Examp_classification_fashionMnist.ipynb.py"})//이미지타이틀


d1.set_fill("회귀 모델의 softmax 다중 분류 구현")//사용자 에필로그
data_sets.push(d2)

// menu3 =============================================================
let d3 = new DataSet("서버프로그램구현")//메인타이틀

data_sets.push(d3)

// menu4 =============================================================
let d4 = new DataSet("배치프로그램구현")//메인타이틀

data_sets.push(d4)
