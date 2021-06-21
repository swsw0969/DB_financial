import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pykrx.stock as stock
path = 'C:\\Users\\SW\\Desktop\\emp\\경제변수.xlsx'
path2 = 'C:\\Users\\SW\\Desktop\\emp\\ISM-MAN_PMI.xlsx'
df_f = pd.read_excel('C:\\Users\\SW\\Desktop\\sample.xlsx', sheet_name = 'Sheet2', index_col = 0)
#일드커브
yield_curve = pd.read_excel(path, Sheet_name= 'yield curve', index_col=0)
#코스피지수
kospi = pd.read_excel(path, sheet_name= 'kospi', index_col=0)
kospi_df=pd.DataFrame(index=yield_curve.index[:])
kospi_df['지수'] = kospi['현재지수']
#원달러환율
exchange_rate = pd.read_excel(path, sheet_name= '원달러환율', index_col=0)
#중국경기선행지수
cli = pd.read_excel(path, sheet_name= '중국CLI', index_col=0)
#TED스프레드
ted = pd.read_excel(path, sheet_name= 'TedRate', index_col=0)
ted = ted.fillna(method = 'pad')
ted.index = ted['날짜']
del ted['날짜']
#무역지수
trade = pd.read_excel(path, sheet_name = '무역수지', index_col=0)
trade = trade.fillna(method = 'pad')
#PMI
pmi = pd.read_excel(path2, Sheet_name = '데이터', index_col=0)
pmi.index = pmi['날짜']
del pmi['날짜'], pmi['MONTH']
#설비투자지표
infra = pd.read_excel(path, sheet_name = '설비투자', index_col = 0)
infra = infra.fillna(method = 'pad')
#장단기스프레드(10-2)
spread = pd.read_excel(path, sheet_name = '장단기', index_col = 0)
spread.index = spread['날짜']
del spread['날짜']
spread = spread.replace('.', 'nan')
spread = spread.astype(np.float64)
#변수 종합데이터프레임
v_sum_df_first = pd.concat([yield_curve, kospi_df, exchange_rate, cli, ted, trade, pmi, infra, spread], axis = 1)
v_sum_df_first = v_sum_df_first.dropna(axis = 0)
v_sum_df_first = v_sum_df_first.loc['2011/03/02':]

# v_sum_df_first = pd.read_excel("C:\\Users\\SW\\Desktop\\emp\\vsum.xlsx", index_col=0)

    ##################벤치마크 및 투자ETF 가져오기######################
    #Kospi 일별 지수화(2006.12.22~20.11.20)


Kospi_index = stock.get_market_ohlcv_by_date("20061222", "20201120", 'kospi', freq='d')[['종가']]
Kospi_index_df = stock.get_market_ohlcv_by_date("20061222", "20201120", 'kospi', freq='d')[['종가']]
Kospi_index.index = Kospi_index.index.strftime('%Y-%m-%d')
v_sum_df_first.index = pd.to_datetime(arg=v_sum_df_first.index, format='%Y-%m-%d')


etf_path = 'C:\\Users\\SW\\Desktop\\emp\\ETF1.xlsx'
etf = pd.read_excel(etf_path, sheet_name = 'ETF', index_col = 0)
etf_list = pd.DataFrame(columns = etf.columns, index = ["weight"])
etf3 = pd.DataFrame(columns = etf.columns,index = ["weight"])

################################################################

from sklearn.preprocessing import StandardScaler
x = v_sum_df_first.values
#scaling
x = StandardScaler().fit_transform(x)
features = ['통안증권(1년)', '국고채(3년)', '국고채(5년)', '국고채(10년)', '지수', '원-미국달러', '중국 CLI',
       'TEDRATE', '무역수지', 'PMI', '설비투자', 'T10Y2Y']
#pca analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=12)
principalComponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data=principalComponents, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9','pc10','pc11','pc12'])
##pca.explained_variance_ratio_
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
pca_var = pd.DataFrame(index = principalDF.columns, columns = ['var', 'cumulative var'])
for i in range(len(per_var)):
    pca_var['var'][i] = per_var[i]
for i in range(len(pca_var)):
    pca_var['cumulative var'][i] = sum(pca_var['var'][:i+1])

#bar plot
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

#pc7개만 쓸거임 96.9%

#Scree Plot
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.plot(pca.explained_variance_, 'o-')

explained_variance_ratio = pca.explained_variance_ratio_
def explained_variance_ratio_plot(explained_variance_ratio1):
    x_axis = range(1, len(explained_variance_ratio1)+1)
    plt.bar(x_axis, explained_variance_ratio1,
            align = 'center', label = 'Individual Explained Variance Ratio')
    plt.step(x_axis, np.cumsum(explained_variance_ratio1),
             where = 'mid', color='red', label='Cumulative Explained Variance Ratio')
    plt.ylim(0, 1.1)
    plt.xticks(x_axis)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid()
    plt.show()   
    
explained_variance_ratio_plot(explained_variance_ratio)

#%%

def backtest(etf, money, start, rebalancing, count):
#    count =0
#    rebalancing =20
    global etf3
    etf2 = pd.DataFrame(columns = etf.columns, index = ["weight"])
    v_sum_df = v_sum_df_first.iloc[:952 + rebalancing * count]
    
    pc1_vect_df = pd.DataFrame(index=v_sum_df.index[119:-21])  #-1
    pc1_vect_list = []
    for i in range(len(v_sum_df.index[119:-21])):          #-1
        pc1_vect_list.append(np.array(principalDF['pc1'][i:i+120]))
    pc1_vect_df['관측벡터'] = pc1_vect_list
    pc1_vect = np.array(pc1_vect_list)
    pc1_var = np.nanvar(pc1_vect)    
    new_pc1 = pc1_vect[-1]
    pc1_distance = []
    for i in range(len(pc1_vect_df.index)):
        past_pc1 = pc1_vect[i]
        pc1_distance.append((sum((new_pc1 - past_pc1)**2/pc1_var)/120)**(1/2))
    pc1_corr = []
    for i in range(len(pc1_vect_df.index)):
        pc1_corr.append(np.corrcoef(pc1_vect[i], pc1_vect[-1])[0,1])
    pc1_adj_distance=[]
    for i in range(len(pc1_vect_df.index)):
        pc1_adj_distance.append(pc1_distance[i] + (1-pc1_corr[i]))
    pc1_vect_df['조정거리'] = pc1_adj_distance
    plt.plot(pc1_vect_df['조정거리'])
    
    
    #PC2 조정거리구하기
    pc2_vect_df = pd.DataFrame(index=v_sum_df.index[119:-21])  #-1
    pc2_vect_list = []
    for i in range(len(v_sum_df.index[119:-21])):          #-1
        pc2_vect_list.append(np.array(principalDF['pc2'][i:i+120]))
    pc2_vect_df['관측벡터'] = pc2_vect_list
    pc2_vect = np.array(pc2_vect_list)
    pc2_var = np.nanvar(pc2_vect)    
    new_pc2 = pc2_vect[-1]
    pc2_distance = []
    for i in range(len(pc2_vect_df.index)):
        past_pc2 = pc2_vect[i]
        pc2_distance.append((sum((new_pc2 - past_pc2)**2/pc2_var)/120)**(1/2))
    pc2_corr = []
    for i in range(len(pc2_vect_df.index)):
        pc2_corr.append(np.corrcoef(pc2_vect[i], pc2_vect[-1])[0,1])
    pc2_adj_distance=[]
    for i in range(len(pc2_vect_df.index)):
        pc2_adj_distance.append(pc2_distance[i] + (1-pc2_corr[i]))
    pc2_vect_df['조정거리'] = pc2_adj_distance
    plt.plot(pc2_vect_df['조정거리'])    
        #PC3 조정거리구하기
    pc3_vect_df = pd.DataFrame(index=v_sum_df.index[119:-21])  #-1
    pc3_vect_list = []
    for i in range(len(v_sum_df.index[119:-21])):          #-1
        pc3_vect_list.append(np.array(principalDF['pc3'][i:i+120]))
    pc3_vect_df['관측벡터'] = pc3_vect_list
    pc3_vect = np.array(pc3_vect_list)
    pc3_var = np.nanvar(pc3_vect)    
    new_pc3 = pc3_vect[-1]
    pc3_distance = []
    for i in range(len(pc3_vect_df.index)):
        past_pc3 = pc3_vect[i]
        pc3_distance.append((sum((new_pc3 - past_pc3)**2/pc3_var)/120)**(1/2))
    pc3_corr = []
    for i in range(len(pc3_vect_df.index)):
       pc3_corr.append(np.corrcoef(pc3_vect[i], pc3_vect[-1])[0,1])
    pc3_adj_distance=[]
    for i in range(len(pc3_vect_df.index)):
        pc3_adj_distance.append(pc3_distance[i] + (1-pc3_corr[i]))
    pc3_vect_df['조정거리'] = pc3_adj_distance
    plt.plot(pc3_vect_df['조정거리'])        
            #PC4 조정거리구하기
    pc4_vect_df = pd.DataFrame(index=v_sum_df.index[119:-21])  #-1
    pc4_vect_list = []
    for i in range(len(v_sum_df.index[119:-21])):          #-1
        pc4_vect_list.append(np.array(principalDF['pc4'][i:i+120]))
    pc4_vect_df['관측벡터'] = pc4_vect_list
    pc4_vect = np.array(pc4_vect_list)
    pc4_var = np.nanvar(pc4_vect)    
    new_pc4 = pc4_vect[-1]
    pc4_distance = []
    for i in range(len(pc4_vect_df.index)):
        past_pc4 = pc4_vect[i]
        pc4_distance.append((sum((new_pc4 - past_pc4)**2/pc4_var)/120)**(1/2))
    pc4_corr = []
    for i in range(len(pc4_vect_df.index)):
        pc4_corr.append(np.corrcoef(pc4_vect[i], pc4_vect[-1])[0,1])
    pc4_adj_distance=[]
    for i in range(len(pc4_vect_df.index)):
        pc4_adj_distance.append(pc4_distance[i] + (1-pc4_corr[i]))
    pc4_vect_df['조정거리'] = pc4_adj_distance
    plt.plot(pc4_vect_df['조정거리'])   
             #PC5 조정거리구하기
    pc5_vect_df = pd.DataFrame(index=v_sum_df.index[119:-21])  #-1
    pc5_vect_list = []
    for i in range(len(v_sum_df.index[119:-21])):          #-1
        pc5_vect_list.append(np.array(principalDF['pc5'][i:i+120]))
    pc5_vect_df['관측벡터'] = pc5_vect_list
    pc5_vect = np.array(pc5_vect_list)
    pc5_var = np.nanvar(pc5_vect)    
    new_pc5 = pc5_vect[-1]
    pc5_distance = []
    for i in range(len(pc5_vect_df.index)):
        past_pc5 = pc5_vect[i]
        pc5_distance.append((sum((new_pc5 - past_pc5)**2/pc5_var)/120)**(1/2))
    pc5_corr = []
    for i in range(len(pc5_vect_df.index)):
        pc5_corr.append(np.corrcoef(pc5_vect[i], pc5_vect[-1])[0,1])
    pc5_adj_distance=[]
    for i in range(len(pc5_vect_df.index)):
        pc5_adj_distance.append(pc5_distance[i] + (1-pc5_corr[i]))
    pc5_vect_df['조정거리'] = pc5_adj_distance
    plt.plot(pc5_vect_df['조정거리'])        
            #PC6 조정거리구하기
    pc6_vect_df = pd.DataFrame(index=v_sum_df.index[119:-21])  #-1
    pc6_vect_list = []
    for i in range(len(v_sum_df.index[119:-21])):          #-1
        pc6_vect_list.append(np.array(principalDF['pc6'][i:i+120]))
    pc6_vect_df['관측벡터'] = pc6_vect_list
    pc6_vect = np.array(pc6_vect_list)
    pc6_var = np.nanvar(pc6_vect)    
    new_pc6 = pc6_vect[-1]
    pc6_distance = []
    for i in range(len(pc6_vect_df.index)):
        past_pc6 = pc6_vect[i]
        pc6_distance.append((sum((new_pc6 - past_pc6)**2/pc6_var)/120)**(1/2))
    pc6_corr = []
    for i in range(len(pc6_vect_df.index)):
        pc6_corr.append(np.corrcoef(pc6_vect[i], pc6_vect[-1])[0,1])
    pc6_adj_distance=[]
    for i in range(len(pc6_vect_df.index)):
        pc6_adj_distance.append(pc6_distance[i] + (1-pc6_corr[i]))
    pc6_vect_df['조정거리'] = pc6_adj_distance
    plt.plot(pc6_vect_df['조정거리'])        
            #PC7 조정거리구하기
    pc7_vect_df = pd.DataFrame(index=v_sum_df.index[119:-21])  #-1
    pc7_vect_list = []
    for i in range(len(v_sum_df.index[119:-21])):          #-1
        pc7_vect_list.append(np.array(principalDF['pc7'][i:i+120]))
    pc7_vect_df['관측벡터'] = pc7_vect_list
    pc7_vect = np.array(pc7_vect_list)
    pc7_var = np.nanvar(pc7_vect)    
    new_pc7 = pc7_vect[-1]
    pc7_distance = []
    for i in range(len(pc7_vect_df.index)):
        past_pc7 = pc7_vect[i]
        pc7_distance.append((sum((new_pc7 - past_pc7)**2/pc7_var)/120)**(1/2))
    pc7_corr = []
    for i in range(len(pc7_vect_df.index)):
        pc7_corr.append(np.corrcoef(pc7_vect[i], pc7_vect[-1])[0,1])
    pc7_adj_distance=[]
    for i in range(len(pc7_vect_df.index)):
        pc7_adj_distance.append(pc7_distance[i] + (1-pc7_corr[i]))
    pc7_vect_df['조정거리'] = pc7_adj_distance
    plt.plot(pc7_vect_df['조정거리'])           
        #    pc_wgt = []
        #    for i in range(len(pca_var.index[:10])):
        #        pc_wgt.append(pca_var['var'][i]/sum(pca_var['var'][:10]))
            
    avg_adj_distance = pd.DataFrame(index=v_sum_df.index)
    avg_adj_distance['pc1'] = pc1_vect_df['조정거리']
    avg_adj_distance['pc2'] = pc2_vect_df['조정거리']
    avg_adj_distance['pc3'] = pc3_vect_df['조정거리']
    avg_adj_distance['pc4'] = pc4_vect_df['조정거리']
    avg_adj_distance['pc5'] = pc5_vect_df['조정거리']
    avg_adj_distance['pc6'] = pc6_vect_df['조정거리']
    avg_adj_distance['pc7'] = pc7_vect_df['조정거리']
    avg_adj_distance = avg_adj_distance.dropna(axis=0)
    avg_adj_distance['거리 합'] = avg_adj_distance.sum(axis=1)
    avg_adj_distance['평균거리'] = avg_adj_distance['거리 합']/7
        #    pct_wgt_df = pd.DataFrame(data = pc_wgt, index = avg_adj_distance.columns)
        #    pct_wgt_df = pct_wgt_df.T
        #    pct_wgt_df_vect = np.array(pct_wgt_df)
        #    avg_adj_distance_vect = np.array(avg_adj_distance)
        #    final_distance = (avg_adj_distance_vect * (1/pct_wgt_df_vect))/10  ##가중치의 역수를 곱해줌(효과: 가중치 큰게 거리가 더 작아짐,즉 더 유사함)
        #    final_distance_df = pd.DataFrame(data=final_distance, index=avg_adj_distance.index, columns=avg_adj_distance.columns)
        #    final_distance_df['최종거리'] = final_distance_df.sum(axis=1)    
    similar_past = avg_adj_distance.sort_values(by='평균거리', ascending=True).head(188)
    similar_past = similar_past.drop(['pc1', 'pc2', 'pc3', 'pc4',
                                              'pc5', 'pc6', 'pc7'], axis = 1)
    
#    pc_wgt = []
#    for i in range(len(pca_var.index[:7])):
#         pc_wgt.append(pca_var['var'][i]/sum(pca_var['var'][:7]))
#    
#    pct_wgt_df = pd.DataFrame(data = pc_wgt, index = avg_adj_distance.columns)
#    pct_wgt_df = pct_wgt_df.T
#    pct_wgt_df_vect = np.array(pct_wgt_df)
##    avg_adj_distance_vect = np.array(avg_adj_distance)
 #   final_distance = (avg_adj_distance_vect * (1/pct_wgt_df_vect)) /7  ##가중치의 역수를 곱해줌(효과: 가중치 큰게 거리가 더 작아짐,즉 더 유사함)
 #   final_distance_df = pd.DataFrame(data=final_distance, index=avg_adj_distance.index, columns=avg_adj_distance.columns)
 #   final_distance_df['최종거리'] = final_distance_df.sum(axis=1)    
    
#    similar_past = final_distance_df.sort_values(by='평균거리', ascending=True).head(188)
#    similar_past = similar_past.drop(['pc1', 'pc2', 'pc3', 'pc4',
 #                                             'pc5', 'pc6', 'pc7'], axis = 1)


            #유사시점에서의 팩터 수익률
    def change21(a, b):
        result = (b/a) - 1
        return result    
    
    
    return_21_df = pd.DataFrame(index = similar_past.index)
    kodex200 = []
    kodexchip = [] 
    kodexcar = []
    kodexbank = []
    kodexinverse = []
    kodexgold = []
    kodexchina = []
    tiger_nasdaq = []
    tiger_b3 = []
    tiger_oil = []
    kosef_dollar = []
            
    index_etf = etf.index     
    index_etf = index_etf.tolist()
            
    similar_index = similar_past.index
    similar_index = similar_index.tolist()
            
    for j in similar_index:
        a = index_etf.index(j)
        kodex200.append(change21(etf['KODEX 200'].iloc[a],etf['KODEX 200'].iloc[a+21]))    
    return_21_df['KODEX 200'] = kodex200
        
    for j in similar_index:
        a = index_etf.index(j)  
        kodexchip.append(change21(etf['KODEX 반도체'].iloc[a],etf['KODEX 반도체'].iloc[a+21]))        
    return_21_df['KODEX 반도체'] = kodexchip
            
    for j in similar_index:
        a = index_etf.index(j)
        kodexbank.append(change21(etf['KODEX 은행'].iloc[a],etf['KODEX 은행'].iloc[a+21]))        
    return_21_df['KODEX 은행'] = kodexbank
        
    for j in similar_index:
        a = index_etf.index(j)
        kodexcar.append(change21(etf['KODEX 자동차'].iloc[a],etf['KODEX 자동차'].iloc[a+21]))        
    return_21_df['KODEX 자동차'] = kodexcar
            
    for j in similar_index:
        a = index_etf.index(j)
        kodexinverse.append(change21(etf['KODEX 인버스'].iloc[a],etf['KODEX 인버스'].iloc[a+21]))        
    return_21_df['KODEX 인버스'] = kodexinverse
        
    for j in similar_index:
        a = index_etf.index(j)
        kodexchina.append(change21(etf['KODEX China H'].iloc[a],etf['KODEX China H'].iloc[a+21]))        
    return_21_df['KODEX China H'] = kodexchina   
            
    for j in similar_index:
        a = index_etf.index(j)
        kodexgold.append(change21(etf['KODEX 골드선물(H)'].iloc[a],etf['KODEX 골드선물(H)'].iloc[a+21]))        
    return_21_df['KODEX 골드선물(H)'] = kodexgold
                
    for j in similar_index:
        a = index_etf.index(j)
        tiger_nasdaq.append(change21(etf['TIGER 미국나스닥100'].iloc[a],etf['TIGER 미국나스닥100'].iloc[a+21]))        
    return_21_df['TIGER 미국나스닥100'] = tiger_nasdaq
        
    for j in similar_index:
        a = index_etf.index(j)
        tiger_b3.append(change21(etf['TIGER 국채3년'].iloc[a],etf['TIGER 국채3년'].iloc[a+21]))        
    return_21_df['TIGER 국채3년'] = tiger_b3
                
    for j in similar_index:
        a = index_etf.index(j)
        tiger_oil.append(change21(etf['TIGER 원유선물Enhanced(H)'].iloc[a],etf['TIGER 원유선물Enhanced(H)'].iloc[a+21]))        
    return_21_df['TIGER 원유선물Enhanced(H)'] = tiger_oil
        
    for j in similar_index:
        a = index_etf.index(j)
        kosef_dollar.append(change21(etf['KOSEF 미국달러선물'].iloc[a],etf['KOSEF 미국달러선물'].iloc[a+21]))        
    return_21_df['KOSEF 미국달러선물'] = kosef_dollar   
                
    Kospi_index_list = []
            
            #코스피 유사시점 수익률
    Kospi_21_return = pd.DataFrame(index = similar_past.index)
    Kospi_index_list = []
            
    for j in similar_index:
        a = index_etf.index(j)
        Kospi_index_list.append(change21(Kospi_index_df['종가'].iloc[a],Kospi_index_df['종가'].iloc[a+21]))
                
    Kospi_21_return['코스피'] = Kospi_index_list
            
            
            #유사기간 팩터별 상대수익률(팩터수익률-코스피수익률)
    return_rel = pd.DataFrame(index = return_21_df.index, columns = return_21_df.columns)
    for i in return_rel.columns:
        return_rel[i][:] = return_21_df[i][:] - Kospi_21_return['코스피'][:]
            ################가중평균 수익률 및 투자비중 결정#########################
    etf_wgt_return = pd.DataFrame(index = return_rel.columns, columns = ['가중평균 수익률', '투자비중'])
    invest_etf = []
            
    for i in return_rel.columns:
        etf_wgt_return['가중평균 수익률'][i] = sum( ((1/similar_past['평균거리'])**3) * return_rel[i] )/ sum( (1/similar_past['평균거리'])**3)
        if etf_wgt_return['가중평균 수익률'][i] > 0:
            invest_etf.append(i)
        elif etf_wgt_return['가중평균 수익률'][i] <= 0:
                pass
    for k in invest_etf:
        etf_wgt_return['투자비중'][k] = etf_wgt_return['가중평균 수익률'][k] / sum(etf_wgt_return['가중평균 수익률'][invest_etf])
    
    if len(invest_etf) > 0:
        print(str(v_sum_df.index[-1])+'에 투자할 ETF는 ' + str(len(invest_etf)) +'종목이며 종목 및 비중(%)은 다음과 같습니다.\n' 
              + str((etf_wgt_return['투자비중'][invest_etf])*100))
        select_num = 1
        
    else :
        select_num = 2        
        
    
    
    etf_wgt_return = etf_wgt_return[etf_wgt_return['투자비중'] > 0]
    invest_etf = etf_wgt_return.index
    invest_weight = etf_wgt_return['투자비중']
    invest_weight = pd.DataFrame(invest_weight)
    
    for x in etf2.columns:
        for y,l in enumerate(invest_weight.index):
            if x == l:
                etf2[x].iloc[0] = invest_weight['투자비중'].iloc[y]
                etf2 = etf2.fillna(0)
##################################
   

    if select_num == 1:
    
 # 아래 주석부분은 TDF 채권 비중 눌려주기 ##########
       
        if v_sum_df.index[-1].year == 2015:
            etf2 = etf2 * 0.8
            etf2["TIGER 국채3년"] += 0.2
                
        if v_sum_df.index[-1].year == 2016:
            etf2 = etf2 * 0.79
            etf2["TIGER 국채3년"] += 0.21
    
        if v_sum_df.index[-1].year == 2017:
            etf2 = etf2 * 0.78
            etf2["TIGER 국채3년"] += 0.22
    
        if v_sum_df.index[-1].year == 2018:
            etf2 = etf2 * 0.77
            etf2["TIGER 국채3년"] += 0.23
    
        if v_sum_df.index[-1].year == 2019:
            etf2 = etf2 * 0.76
            etf2["TIGER 국채3년"] += 0.24
            
        if v_sum_df.index[-1].year == 2020:
            etf2 = etf2 * 0.75
            etf2["TIGER 국채3년"] += 0.25
    
        if v_sum_df.index[-1].year == 2021:
            etf2 = etf2 * 0.74
            etf2["TIGER 국채3년"] += 0.26
        
        etf2.index = [str(v_sum_df.index[-1])]
        etf3 = pd.concat([etf3, etf2])

        for i in etf2.columns:
            if etf2[i].iloc[0] == 0:
                del etf2[i]
                
        invest_list = etf2.columns

        etf_portfolio = etf[invest_list][start:]
    
        
        print("--------------------")
        print('날짜 : {}'.format(date))
        print("백테시작")
        
    
        pf_stock_num = {}
        stock_amount = 0
                                
        each_money_list = []
    
        for i in etf2.iloc[0]:
            each_money = money * i
            each_money_list.append(each_money)
    
        for i, code in enumerate(etf_portfolio.columns):
            temp = int(each_money_list[i] / etf_portfolio[code][0]) # 몇개 살건데?
            pf_stock_num[code] = temp
            stock_amount = stock_amount + temp * etf_portfolio[code][0]
    
                
                
        cash_amount = money - stock_amount
                    
                    
        stock_pf = 0
                        
        for code in etf_portfolio.columns:
            stock_pf = stock_pf + etf_portfolio[code] * pf_stock_num[code]
                    
        back = pd.DataFrame({'종목':stock_pf[:rebalancing]})
                        
        back['현금'] = [cash_amount] * len(back)
                    
        back['종합'] =back['종목'] + back['현금']
                    
        back['일별수익률'] = back['종합'].pct_change()
        back['총수익률'] = back['종합']/money - 1
                    
        money = back.iloc[-1,2]
                    
    
        invest = etf_portfolio.columns
                
        list_mmt = pd.DataFrame(invest, columns = [date])
        list_mmt = list_mmt.T
        
        money = money * (1-0.0025)
        count = count + 1
        print("백테 결과 남은돈 : ", money)

        return back, money, list_mmt, count
    
    
    
 ## 아래 주석부분은 국면인식 비중으로만! ##########

        # etf_portfolio = etf[invest_etf][start:]
    
        
        # print("--------------------")
        # print('날짜 : {}'.format(date))
        # print("백테시작")
        
    
        # pf_stock_num = {}
        # stock_amount = 0
                                
        # each_money_list = []
    
        # for i in invest_weight["투자비중"]:
        #     each_money = money * i
        #     each_money_list.append(each_money)
    
        # for i, code in enumerate(etf_portfolio.columns):
        #     temp = int(each_money_list[i] / etf_portfolio[code][0]) # 몇개 살건데?
        #     pf_stock_num[code] = temp
        #     stock_amount = stock_amount + temp * etf_portfolio[code][0]
    
                
                
        # cash_amount = money - stock_amount
                    
                    
        # stock_pf = 0
                        
        # for code in etf_portfolio.columns:
        #     stock_pf = stock_pf + etf_portfolio[code] * pf_stock_num[code]
                    
        # back = pd.DataFrame({'종목':stock_pf[:rebalancing]})
                        
        # back['현금'] = [cash_amount] * len(back)
                    
        # back['종합'] =back['종목'] + back['현금']
                    
        # back['일별수익률'] = back['종합'].pct_change()
        # back['총수익률'] = back['종합']/money - 1
                    
        # money = back.iloc[-1,2]
                    
    
        # invest = etf_portfolio.columns
                
        # list_mmt = pd.DataFrame(invest, columns = [date])
        # list_mmt = list_mmt.T
        
        # money = money * (1-0.0025)
        # count = count + 1
        # print("백테 결과 남은돈 : ", money)

        # return back, money, list_mmt, count


    if select_num == 2:

####### 이부분은 현금보유일때 다른 자산 투자 #######################
    
        etf2 = etf2.T
        etf2["weight"] = 0
        etf2 = etf2.T
        
        etf2["TIGER 국채3년"] = 1
        # etf2["KODEX 골드선물(H)"] = 1
        
        etf2.index = [str(v_sum_df.index[-1])]
        
        etf3 = pd.concat([etf3, etf2])

        etf2 = etf2[["TIGER 국채3년"]]
        invest_list = etf2.columns

        etf_portfolio = etf[invest_list][start:]

        print("--------------------")
        print('날짜 : {}'.format(date))
        print("백테시작")
        
    
        pf_stock_num = {}
        stock_amount = 0
                                
        each_money_list = []
    
        for i in etf2.iloc[0]:
            each_money = money * i
            each_money_list.append(each_money)
     
        for i, code in enumerate(etf_portfolio.columns):
            temp = int(each_money_list[i] / etf_portfolio[code][0]) # 몇개 살건데?
            pf_stock_num[code] = temp
            stock_amount = stock_amount + temp * etf_portfolio[code][0]
    
                
                
        cash_amount = money - stock_amount
                    
                    
        stock_pf = 0
                        
        for code in etf_portfolio.columns:
            stock_pf = stock_pf + etf_portfolio[code] * pf_stock_num[code]
                    
        back = pd.DataFrame({'종목':stock_pf[:rebalancing]})
                        
        back['현금'] = [cash_amount] * len(back)
                    
        back['종합'] =back['종목'] + back['현금']
                    
        back['일별수익률'] = back['종합'].pct_change()
        back['총수익률'] = back['종합']/money - 1
                    
        money = back.iloc[-1,2]
                    
    
        invest = etf_portfolio.columns
                
        list_mmt = pd.DataFrame(invest, columns = [date])
        list_mmt = list_mmt.T
        
        money = money * (1-0.0025)
        count = count + 1
        print("백테 결과 남은돈 : ", money)

        return back, money, list_mmt, count

######################################### 아래는 현금보유 #######################


        # print("--------------------")
        # print('날짜 : {}'.format(date))
        # print('돈')
        # print("백테 X")
        # etf_portfolio = etf['KODEX 200'][start:]
    
                    
                
        # back = pd.DataFrame({'종목':etf_portfolio[:rebalancing]})
        # back['종목'] = 0
        # back['현금'] = money
        # back['종합'] = back['종목'] + back['현금']
        # back['일별수익률'] = back['종합'].pct_change()
        # back['총수익률'] = back['종합']/money - 1
                                        
        # list_mmt = None
                        
        # count = count + 1   
        # return back, money, list_mmt, count   

#%%

back_list = []
start = '2015-03-02'
endday = '2020-11-20'
money = 100000000
back = None
backsum = None
list_mmt = None
list_sum = None
plot = 20
count = 0
rebalancing = 20

#%%    
for date, i in zip(etf[start:endday].index, range(len(etf[start:endday].index))):
    
    if i % plot == 0:
        back, money, list_mmt, count = backtest(etf, money, date, plot, count)
        
        if i == 0 :
            backsum = back
            list_sum = list_mmt
                
        else:
                
            backsum = pd.concat([backsum, back])
            list_sum = pd.concat([list_sum, list_mmt])

    if i == (len(etf[start:endday].index) - 1):
        backsum = backsum.loc[:endday]
        back_list.append(backsum)
            
        # if i == (len(data[start:endday].index) - 1):
        #     backsum = backsum.loc[:endday]
        #     back_list.append(backsum)            
            
back_list = back_list[0]

#%%

import pykrx.stock as stock
from pykrx import stock
import datetime

index_start = start.replace('-',"")
index_endday = endday.replace('-',"")

kospi = stock.get_market_ohlcv_by_date(index_start, index_endday, 'kospi', freq='d')[['종가']]
kosdaq = stock.get_market_ohlcv_by_date(index_start, index_endday, 'kosdaq', freq='d')[['종가']]

kospi.columns = ['KOSPI']
kosdaq.columns = ['KOSDAQ']

kospi['총수익률'] = (kospi['KOSPI'] / kospi['KOSPI'][0]) - 1
kosdaq['총수익률'] = (kosdaq['KOSDAQ'] / kosdaq['KOSDAQ'][0]) - 1

plt.figure(figsize=(15,6))

back_list['일별수익률'] = back_list['일별수익률'].pct_change()
back_list['총수익률'] = back_list['종합'] / back_list['종합'][0]- 1
    
    
    
kosdaq.index= back_list.index
kospi.index = back_list.index
kosdaq['총수익률'].plot(label = 'kosdaq')
kospi['총수익률'].plot(label = 'kospi')
back_list['총수익률'].plot(label = 'portfoilo')
plt.title("PortFolio")
plt.legend()
plt.show()
        
#%%

test_set = back_list

test_set['daily_rtn'] = test_set['종합'].pct_change()
test_set['st_rtn'] = (1+test_set['daily_rtn']).cumprod()

CAGR_list = []
MDD_list = []
VOL_list = []
Sharpe_list= []
historical_dd_list = []


CAGR = test_set.loc[endday,'st_rtn'] ** (252./len(test_set.index)) - 1
    
historical_max = test_set['종합'].cummax()
daily_drawdown = test_set['종합'] / historical_max - 1.0
historical_dd = daily_drawdown.cummin()
MDD = historical_dd.min()
VOL = np.std(test_set['daily_rtn']) * np.sqrt(252.)
Sharpe = np.mean(test_set['daily_rtn']) / np.std(test_set['daily_rtn']) * np.sqrt(252.)
    
CAGR = str(round(CAGR*100,2)) + '%'
Sharpe = round(Sharpe,2)
VOL = str(round(VOL*100,2)) + '%'
MDD = str(round(-1*MDD*100,2)) +'%'
    
    
CAGR_list.append(CAGR)
MDD_list.append(MDD)
VOL_list.append(VOL)
Sharpe_list.append(Sharpe)
historical_dd_list.append(historical_dd)

b = pd.DataFrame(columns = ('CAGR', 'MDD', 'Sharpe', 'VOL'))

b['CAGR'] = CAGR_list
b['MDD'] = MDD_list
b['Sharpe'] = Sharpe_list
b['VOL'] = VOL_list