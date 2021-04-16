import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
from scipy.stats import zscore
from itertools import combinations
import plotly.graph_objs as go
import plotly.figure_factory as ff

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', low_memory=False)

print(df.head())
# df.info()


def absHighPass(df, absThresh):
    passed = set()
    for (r, c) in combinations(df.columns, 2):
        if abs(df.loc[r, c]) >= absThresh:
            passed.add(r)
            passed.add(c)
    passed = sorted(passed)
    return df.loc[passed, passed]


def hatali_deger_test(a):
    for column_name in a:
        print("{} sütunu için problemli değerler : ".format(column_name))
        hatali_degerler = []
        for deger in df[column_name]:
            try:
                float(deger)
            except:
                print(deger)
                hatali_degerler.append(deger)
        print(set(hatali_degerler))


def churn_kategorize_et(value):
    if value == "No":
        return 0
    elif value == "Yes":
        return 1

def gender_kategorize_et(value):
    if value == "Female":
        return 0
    else:
        return 1


def plot_distribution(var_select, bin_size):
    tmp1 = churn[var_select]
    tmp2 = no_churn[var_select]
    hist_data = [tmp1, tmp2]

    group_labels = ['Churn : yes', 'Churn : no']
    colors = ['gold', 'lightblue']

    fig = ff.create_distplot(hist_data, group_labels, colors=colors, show_hist=True, curve_type='kde', bin_size=bin_size)

    fig['layout'].update(title=var_select, autosize=False, height=500, width=800)

    py.plot(fig, filename='Density plot')


# Null olan degerlerin sayisnin listelenmesi (veri setimizde boyle degerler oldugu gozukmuyor )

print(df.isna().sum())

# Her sutundaki benzersiz degerleri goruntuleme

print(df.nunique())

for sutun_adi in df.columns:
    print("{} sütunundaki benzersiz değerler : {}".format(sutun_adi, df[sutun_adi].unique()))

# Sutunlarin veri tiplerini inceleme

print(df.dtypes)

hatali_deger_test(["TotalCharges","tenure"])

# 0 ay abone olan musterilerin toplam odedigi ucret " "  olarak gozukmekte

df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float64')


# Bazi degerlerin kategorize edilmesi kategorize edilmesi

df['Churn-category'] = df['Churn'].apply(churn_kategorize_et)
df['Gender-category'] = df['gender'].apply(gender_kategorize_et)
df['Partner-category'] = df['Partner'].apply(churn_kategorize_et)
df['PhoneService-category'] = df['PhoneService'].apply(churn_kategorize_et)
print(df.dtypes)


# Veriler hakkinda genel bir bilgi olusturmak icin korelasyon matrisimizi olusturuyoruz
# Korelasyon matirisnde goruyoruz ki Aylik ve toplam ucretin serviste kalip kalmamak
# ile ilgili bir baglantisi olabilir

korelasyon_mat_df = df.corr()
print(korelasyon_mat_df)
sns.heatmap(absHighPass(korelasyon_mat_df,0.2), annot=True, cmap="YlGnBu")
plt.title("Korelasyon Matrisi")


# Aylik ucret ve Toplam ucret grafiklerini incelersek

plt.show()

plt.boxplot(df["MonthlyCharges"])
plt.title("Aylik ucret boxplot")
plt.show()

plt.boxplot(df["TotalCharges"])
plt.title("Toplam odenen ucret boxplot")
plt.show()


# Z-score yontemi ile uc deger arama
#Aylik
z_scores = zscore(df["MonthlyCharges"])
for threshold in range(1, 5):
    print("Eşik değeri: {}".format(threshold))
    print("Aylikta aykırı değerlerin sayısı: {}".format(len((np.where(z_scores > threshold)[0]))))
    print('------')
#Toplam
z_scores = zscore(df["TotalCharges"])
for threshold in range(1, 5):
    print("Eşik değeri: {}".format(threshold))
    print("Toplamda aykırı değerlerin sayısı: {}".format(len((np.where(z_scores > threshold)[0]))))
    print('------')

# Tukey yontemi
#Aylik
print("Aylik Tukey")
q75, q25 = np.percentile(df["MonthlyCharges"], [75, 25])
caa = q75 - q25
esik_degerleri = pd.DataFrame()
for esik_degeri in np.arange(1, 5, 0.5):
    min_deger = q25 - (caa * esik_degeri)
    maks_deger = q75 + (caa * esik_degeri)
    aykiri_deger_sayisi = len((np.where((df["MonthlyCharges"] > maks_deger) |
                                        (df["MonthlyCharges"] < min_deger))[0]))
    esik_degerleri = esik_degerleri.append({'esik_degeri': esik_degeri, 'aykiri_deger_sayısı': aykiri_deger_sayisi},
                                           ignore_index=True)
print(esik_degerleri)
#Toplam
print("Toplam Tukey")
q75, q25 = np.percentile(df["TotalCharges"], [75, 25])
caa = q75 - q25
esik_degerleri = pd.DataFrame()
for esik_degeri in np.arange(1, 5, 0.5):
    min_deger = q25 - (caa * esik_degeri)
    maks_deger = q75 + (caa * esik_degeri)
    aykiri_deger_sayisi = len((np.where((df["TotalCharges"] > maks_deger) |
                                        (df["TotalCharges"] < min_deger))[0]))
    esik_degerleri = esik_degerleri.append({'esik_degeri': esik_degeri, 'aykiri_deger_sayısı': aykiri_deger_sayisi},
                                           ignore_index=True)
print(esik_degerleri)

#Onemli aykiri deger yok gibi gozukmekte


# Yogunluk grafikleri

#Veri ormal dagilim degil Bi-Modal dagilim gostermistir
#Kullanicilarin cogu 18 ile 25 dolar arasi ucret odemektedir giris seviyesi ana paket olarak gozukmekte
sns.distplot(df["MonthlyCharges"])
plt.show()

#Veri pozitif(saga) carpiktir
#Kullanicilar ortalama 1100 dolar gibi toplam ucret odemislerdir
sns.distplot(df["TotalCharges"])
plt.show()




trace = go.Bar(
    x=(df['Churn'].value_counts().values.tolist()),
    y=['Churn : no', 'Churn : yes'],
    orientation='h', opacity=0.8,
    text=df['Churn'].value_counts().values.tolist(),
    textfont=dict(size=15),
    textposition='auto',
    marker=dict(
        color=['lightblue', 'gold'],
        line=dict(color='#000000', width=1.5)
    ))

layout = dict(title='Ayrilip ayrilmama sayilari',
              autosize=False,
              height=500,
              width=800)

fig = dict(data=[trace], layout=layout)
py.plot(fig)




#Servisten ayrilan kullanicilar genellikle ilk ay kullananlar
#Servisten ayrilma orani kullanma suresi gectikte artmaktadir
#Kullanici 10-20 ay arasi serviste kaldiysa daha uzun sure durmaya egilimlidir
#65 ay ve uzeri uye olanlarin cogu serviste halen kalmaktadir

df['Churn_Num'] = df['Churn'].map( {'Yes': 1, 'No': 0} ).astype(int)
fighist = sns.FacetGrid(df, col='Churn_Num')
fighist.map(plt.hist, 'tenure', bins=20)
plt.show()





# Churn(ayrilma) durumunun toplam abonelik uzunlugu, aylik odenen ucret ve toplam odenmis ucret ile karsilastirilmasi
df.Churn.replace(to_replace=dict(Yes=1, No=0), inplace=True)

col_name = ['SeniorCitizen', 'Churn']
df[col_name] = df[col_name].astype(object)

churn = df[(df['Churn'] != 0)]
no_churn = df[(df['Churn'] == 0)]


plot_distribution('MonthlyCharges', False)
time.sleep(2)
plot_distribution('TotalCharges', False)