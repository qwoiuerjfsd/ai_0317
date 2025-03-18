
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
d1.set_img(0,{imgtitle:"모델의 예측결과 측정",imglog:"테스트 데이터를 주입하여 예측결과를 인출하고 실제 정답과 차이를 ",imgurl:"https://drive.google.com/file/d/1YdeZKYxbEB4IDbzX3_F7u4_mJibbMYvR/view?usp=drive_link",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀




d1.set_content("캘리포니아 주택 가격 예측 선형 회귀모델")
d1.set_img(1,{imgtitle:"캘리포니아 주택 가격 예측 선형 회귀모델",imgurl:"",imglog:"",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀



d1.set_content("당뇨상태 1년후 예측 선형 회귀모델")
d1.set_img(2,{imgtitle:"1년후 당뇨상태 예측",imgurl:"",imglog:"",sourceurl:"https://github.com/qwoiuerjfsd/ai_0317/blob/main/LinearRegression/examp_LinearRegression_bostonHousing.py"})//이미지타이틀

d1.set_fill("선형 회귀 모델은 단일데이터를 이용하거나 다중 데이터를 이용하여 연속적인 값을 출력하여 예측한다")//사용자 에필로그
data_sets.push(d1)

// menu2 =============================================================
let d2 = new DataSet("공통모듈구현")//메인타이틀

data_sets.push(d2)

// menu3 =============================================================
let d3 = new DataSet("서버프로그램구현")//메인타이틀

data_sets.push(d3)

// menu4 =============================================================
let d4 = new DataSet("배치프로그램구현")//메인타이틀

data_sets.push(d4)
