#삶의 다양성이 출산율에 미치는 영향

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


print('이 코드는 OECD 38개 국가 내 삶의 다양성이 출산율에 미치는 영향을 알아보는 것을 목적으로 합니다.')

# [종속변수] 출산율
# 국가별로 2019-2021 3개년 평균을 도출합니다.
fertility_data = pd.read_csv('Fertility rate.csv')
filtered_data = fertility_data[fertility_data['TIME'].isin([2019, 2020, 2021])]
average_fertility = filtered_data.groupby('Country')['Value'].mean()
fertility_dict = average_fertility.to_dict()
print('국가별 2019-2021 3개년 출산율 평균은 다음과 같습니다.')
print('fertility rate is =', fertility_dict)
print('')

## [독립변수] diversity index
## diversity index는 5가지 변수가 조합되어 제작됩니다.

print('출산율에 영향을 미치는 독립변수로 추정되는 "삶의 다양성" 지수는 5가지 변수를 조합해 제작합니다.')
print('')

# [지수의 1번째 변수] 출신 국가에 따른 다양성 지수
# PDF 파일을 바탕으로 별도의 CSV파일을 제작했습니다.
country_diversity_index_data = pd.read_csv('Diversity index based on country of birth 2015.csv')
country_diversity_index_dict = country_diversity_index_data.set_index('Country')['Country of birth diversity index'].to_dict()
print('삶의 다양성 지수의 첫 번째 변수인 출신 국가 다양성 지수는')
print('country diversity index is =', country_diversity_index_dict)
print('')

# [지수의 2번째 변수] 국가별 동거 커플 비율(법률혼 대비 동거 커플 비율)
# CSV 파일에서 OECD 국가를 추가하고 불필요한 데이터를 제거해 별도의 CSV파일을 제작했습니다.
Cohabitation_data = pd.read_csv('Cohabitation partnership ratio.csv')
cohabitation_marriage_dict = {
    row['Country']: row['Cohabiting'] / row['Married or in a civil or registered partnership']
    for index, row in Cohabitation_data.iterrows()
}
print('삶의 다양성 지수의 두 번째 변수인 법률혼 대비 동거 커플 비율은')
print('cohabition partnership ratio is =', cohabitation_marriage_dict)
print('')

# [지수의 3번째 변수] 인구 밀도 (Note. 인구밀도와 삶의 다양성은 '역'의 관계를 지님을 전제합니다.)
# CSV 파일에서 OECD 국가로 한정해 리스트를 수정하고, 국가별로 2014-2016 3개년 평균값을 엑셀 내에서 도출했습니다.
population_density_dict = {"AUS": 3.100287641, "AUT": 104.7250081, "BEL": 372.2443307, "CAN": 3.98746132, "CHE": 209.6029814, "CHL": 24.04788272, "COL": 42.48869761, "CRI": 95.86587675, "CZE": 136.5815092, "DEU": 233.9913234, "DNK": 142.1247333, "ESP": 92.97489511, "EST": 30.25643739, "FIN": 18.02804058, "FRA": 121.4999492, "GBR": 269.1275851, "GRC": 84.01672356, "HUN": 107.8366535, "IRL": 68.29744036, "ISL": 3.303873649, "ISR": 387.2735675, "ITA": 205.3163487, "JPN": 348.8733425, "KOR": 523.2493731, "LTU": 46.32016122, "LUX": 221.138543, "LVA": 31.76994981, "MEX": 61.8028593, "NLD": 503.1703111, "NOR": 14.20460259, "NZL": 17.52053979, "POL": 124.0713653, "PRT": 113.1102247, "SVK": 112.8190682, "SVN": 102.4508704, "SWE": 24.07473672, "TUR": 103.4166396, "USA": 35.06260337}
print('삶의 다양성 지수의 세 번째 변수인 인구밀도는')
print('population density is =', population_density_dict)
print('')

#[지수의 4번째 변수] 종교 다양성 지수
# PDF 파일을 바탕으로 별도의 CSV파일을 제작했습니다.
religious_diversity_data = pd.read_csv('Religious Diversity Index.csv')
religious_diversity_dict = religious_diversity_data.set_index('Country')['Religious Diversity Index'].to_dict()
print('삶의 다양성 지수의 네 번째 변수인 종교 다양성 지수는')
print('religious diversity index is =', religious_diversity_dict)
print('')

#[지수의 5번째 변수] 성정체성 사회 용인 지수
# PDF 파일을 바탕으로 별도의 CSV파일을 제작했습니다.
#"Index on Legal Recognition of Homosexual Orientation"와 "Transgender Rights Index in OECD countries"값을 평균한 값을 도출합니다.
LGBT_legal_recognition_data = pd.read_csv('Index on Legal Recognition of Homosexual Orientation in OECD countries as of 2016.csv')
LGBT_legal_recognition_dict = LGBT_legal_recognition_data.set_index('Country')['Index on Legal Recognition of Homosexual Orientation in OECD countries as of 2016'].to_dict()

Transgender_Rights_index_data = pd.read_csv('Transgender Rights Index in OECD countries, as of 2016.csv')
Transgender_Rights_index_dict = Transgender_Rights_index_data.set_index('Country')['Transgender Rights Index in OECD countries, as of 2016'].to_dict()

#[지수의 5번째 변수] 성정체성 사회 용인 지수
#위 두 index를 평균화함으로써 성정체성 사회용인 지수를 더욱 고도화합니다.
sexual_identity_average_index_dict = {}
for country, index in LGBT_legal_recognition_dict.items():
    if country in Transgender_Rights_index_dict:
        average_index = (index + Transgender_Rights_index_dict[country]) / 2
        sexual_identity_average_index_dict[country] = average_index
print('삶의 다양성 지수의 다섯 번째 변수인 성정체성 사회 용인 지수는')
print('sexual identity index is =', sexual_identity_average_index_dict)
print('')

print('출산율을 제외한 5개의 dictionary에 missing Value가 존재합니다. 이는 정규화를 가로막습니다.')
print('국가를 같은 문화권으로 분류하고, 해당 문화권의 평균 값으로 결측값을 대체합니다.')
print('')

# Cultural regions(출처 보고서 참고)
cultural_regions = {
    'us_canada': ['USA', 'CAN'],
    'middle_south_america': ["CHL", "COL", "MEX", "CRI"],
    'east_southeast_asia': ['KOR', 'JPN'],
    'western_europe': ["AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "GRC", "ISL", "IRL", "ITA", "LUX", "NLD", "NOR", "PRT", "ESP", "SWE", "CHE", "GBR"],
    'eastern_europe_central_eurasia': ["CZE", "EST", "HUN", "LVA", "LTU", "POL", "SVK", "SVN"],
    'australia_pacific_islands': ["AUS", "NZL"],
    'southwest_asia_northern_africa': ["ISR", "TUR"]
}

# missing value를 대체하는 함수를 제작합니다.
def fill_missing_values(data_dict, cultural_regions):
    for region, countries in cultural_regions.items():
        # NaN을 제외한 데이터를 추출합니다.
        region_values = [value for key, value in data_dict.items() if key in countries and not np.isnan(value)]
        # 문화권마다 구분해 평균값을 도출합니다.
        if region_values:
            region_average = np.mean(region_values)
            #  Missing value를 위에서 구한 문화권별 평균값으로 대체합니다.
            for country in countries:
                if country in data_dict and np.isnan(data_dict[country]):
                    data_dict[country] = region_average

# 각각의 dictionary의 missing value를 채워 넣습니다.
fill_missing_values(country_diversity_index_dict, cultural_regions)
fill_missing_values(cohabitation_marriage_dict, cultural_regions)
fill_missing_values(population_density_dict, cultural_regions)
fill_missing_values(religious_diversity_dict, cultural_regions)
fill_missing_values(sexual_identity_average_index_dict, cultural_regions)

print('missing values를 채워 넣은 dictionary는 다음과 같습니다.')
print('country_diversity_index_dict =', country_diversity_index_dict)
print('cohabitation_marriage_dict =', cohabitation_marriage_dict)
print('population_density_dict =', population_density_dict)
print('religious_diversity_dict =', religious_diversity_dict)
print('sexual_identity_average_index_dict =', sexual_identity_average_index_dict)
print('')


# 6개의 dictionary를 정규화합니다.
#정규화 함수를 정의합니다.
def normalize_data(data_dict):
    min_value = min(data_dict.values())
    max_value = max(data_dict.values())
    normalized_dict = {k: (v - min_value) / (max_value - min_value) for k, v in data_dict.items()}
    return normalized_dict

# 정규화를 진행합니다.
print('서로 다른 평균과 표준편차를 지닌 자료이므로 정규화가 필수적입니다.')
print('인구밀도의 경우 최솟값이 1을 갖도록 역으로 정규화합니다.')
print('')
normalized_fertility_rate = normalize_data(fertility_dict)
normalized_country_diversity_index = normalize_data(country_diversity_index_dict)
normalized_cohabitation_marriage = normalize_data(cohabitation_marriage_dict)
normalized_population_density = normalize_data(population_density_dict)
normalized_religious_diversity = normalize_data(religious_diversity_dict)
normalized_sexual_identity = normalize_data(sexual_identity_average_index_dict)

#단, 인구밀도는 높을 수록 삶의 다양성이 낮아지므로 최댓값을 0, 최솟값을 1로 지정해야 합니다.
#정규화를 반대로 하는 함수를 정의합니다.
def invert_normalize_data(data_dict):
    min_value = min(data_dict.values())
    max_value = max(data_dict.values())
    inverted_normalized_dict = {k: (max_value - v) / (max_value - min_value) for k, v in data_dict.items()}
    return inverted_normalized_dict

# 함수를 적용해 정규화를 바로잡습니다.
inverted_normalized_population_density = invert_normalize_data(population_density_dict)

# 정규화된 dictionary를 출력합니다.
print('normalized_fertility_rate =', normalized_fertility_rate)
print('normalized_country_diversity_index =', normalized_country_diversity_index)
print('normalized_cohabitation_marriage =', normalized_cohabitation_marriage)
print('normalized population density =', inverted_normalized_population_density)
print('normalized_religious_diversity =', normalized_religious_diversity)
print('normalized_sexual_identity =', normalized_sexual_identity)
print('')

print("5가지 변수를 조합해 '삶의 다양성 지수(live_diversity_index)'를 제작합니다.")
# life_diversity_index를 정의하는 함수입니다. 단순한 산술평균을 적용합니다.
def calculate_life_diversity_index(norm_dicts):
    life_diversity_index = {}
    for country in norm_dicts[0].keys():
        mean_value = np.mean([d.get(country, np.nan) for d in norm_dicts])
        if not np.isnan(mean_value):
            life_diversity_index[country] = mean_value
    return life_diversity_index

normalized_dictionaries = [
    normalized_country_diversity_index, 
    normalized_cohabitation_marriage, 
    inverted_normalized_population_density, 
    normalized_religious_diversity, 
    normalized_sexual_identity
]

life_diversity_index = calculate_life_diversity_index(normalized_dictionaries)

print('삶의 다양성 지수는 다음과 같습니다.')
print(life_diversity_index)

common_countries = set(life_diversity_index.keys()).intersection(normalized_fertility_rate.keys())
life_diversity_values = [life_diversity_index[country] for country in common_countries]
fertility_rate_values = [normalized_fertility_rate[country] for country in common_countries]
correlation, p_value = pearsonr(life_diversity_values, fertility_rate_values)

slope, intercept = np.polyfit(life_diversity_values, fertility_rate_values, 1)

# 산점도와 regression line을 제작합니다.
plt.figure(figsize=(10, 6))
plt.scatter(life_diversity_values, fertility_rate_values, alpha=0.7)
plt.plot(life_diversity_values, np.polyval([slope, intercept], life_diversity_values), color='red')  # Regression line
plt.title('Scatter Plot of Life Diversity Index vs Normalized Fertility Rate')
plt.xlabel('Life Diversity Index')
plt.ylabel('Normalized Fertility Rate')
plt.grid(True)
plt.show()

print(f'Pearson Correlation Coefficient: {correlation}')
print(f'P-value: {p_value}')
