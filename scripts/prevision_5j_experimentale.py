"""EXPÉRIMENTAL — simulation d'un système de prévision à 5 jours (2017-2025) et comparaison au CEVA.

    python scripts/affiner_s2.py --toutes-claires        # surfaces corrigées de toutes les images claires
    python scripts/prevision_5j_experimentale.py corrige --maree <daily_risk.parquet>

VERSION : brut (surfaces brutes) | corrige (algues fixées exclues).
La houle hivernale a été testée (ERA5) et retirée : aucun effet fiable (voir notebook 05).
La marée prévue (hauteur d'eau à 11 h UTC, marégraphes SHOM) vient pour l'instant du fichier
app/data/daily/daily_risk.parquet de la branche feature/daily-risk d'Allan (--maree) : à remplacer
par un module d'ingestion commun quand les deux branches seront réunies.
Sorties dans data/processed/prevision_5j/.

Pour chaque année Y, le modèle est entraîné sans l'année Y (une année laissée de côté).
À chaque date t (tous les 5 jours, avril-octobre) : dernière image claire ≤ t (≤ 20 j d'âge),
marée prévue à t+5 (hauteur d'eau à 11 h UTC, SHOM via la branche d'Allan), météo/débits/azote ≤ t.
Deux prévisions : surface visible au passage du satellite à t+5 ; surface découverte à basse mer
(hauteur d'eau fixée au 10e centile de la baie) = ce que mesure un survol CEVA.
"""
import sys,warnings,numpy as np,pandas as pd
import argparse
from pathlib import Path
_ap=argparse.ArgumentParser(); _ap.add_argument("version",choices=["brut","corrige"])
_ap.add_argument("--maree",required=True,help="daily_risk.parquet (branche feature/daily-risk)")
_args=_ap.parse_args(); VERSION=_args.version
warnings.filterwarnings("ignore"); sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from algathon.series_annuelles import charger_config, serie_riviere
from algathon.ingestion import meteo, hubeau
S=str(Path(__file__).resolve().parents[1]/"data"/"processed"/"prevision_5j")+"/"; Path(S).mkdir(parents=True,exist_ok=True)
pd.set_option("display.width",230)
conf=charger_config()
# --- marée prévue (hauteur d'eau à 11 h UTC), par baie
al=pd.read_parquet(_args.maree)
al=al[~al.is_forecast].groupby(["sector","date"])[["level_11utc_m","coef"]].first().reset_index()
SECT={"saint_brieuc":"saint_brieuc","lieue_de_greve":"lannion","fresnaye":"fresnaye","horn_guillec":"horn_guillec",
      "guisseny":"guisseny","douarnenez":"douarnenez","concarneau_la_foret":"douarnenez"}  # Concarneau : Brest en approximation
maree={b:al[al.sector==sec].set_index("date")[["level_11utc_m","coef"]] for b,sec in SECT.items()}
# --- images claires
s=pd.read_csv("data/processed/s2_scenes_claires_corrige.csv" if VERSION!="brut" else "data/processed/s2_scenes.csv"); s=s[s.zone_ha>0].copy(); s["pv"]=s.valid_ha/s.zone_ha
c=s[s.pv>=0.9].sort_values("valid_ha").drop_duplicates(["baie","date"],keep="last")
c["date"]=pd.to_datetime(c.date); c=c[c.date>="2017-01-01"].sort_values(["baie","date"]).reset_index(drop=True)
c["ulve"]=c.ulva_corr_s020 if VERSION!="brut" else c.ulva_ha_s020
# --- variables journalières
jour={}
for b,cfg in conf["baies"].items():
    m=meteo.meteo_journaliere(cfg["meteo"]["lat"],cfg["meteo"]["lon"]).set_index("date")
    q=f=None
    for r in cfg["rivieres"]:
        sr=serie_riviere(hubeau.debits_journaliers(r["debit"]),[hubeau.nitrates(x) for x in r["nitrates"]])
        q=sr.debit_m3s if q is None else q.add(sr.debit_m3s); f=sr.flux_kgN if f is None else f.add(sr.flux_kgN)
    jour[b]=m.join(pd.DataFrame({"debit":q,"flux":f}),how="left")
NIV_BAS={b:maree[b].level_11utc_m[maree[b].index.month.isin(range(4,11))].quantile(0.10) for b in maree}

def variables(b,t_img,ulve0,t_cible,niveau_cible=None):
    J=jour[b]; M=maree[b]
    w5=J.loc[t_img-pd.Timedelta(days=5):t_img]; w10=J.loc[t_img-pd.Timedelta(days=10):t_img]; w30=J.loc[t_img-pd.Timedelta(days=30):t_img]
    n1=M.level_11utc_m.get(t_cible,np.nan) if niveau_cible is None else niveau_cible
    return {"l_ulve0":np.log1p(ulve0),"niveau0":M.level_11utc_m.get(t_img,np.nan),"niveau1":n1,
            "coef1":M.coef.get(t_cible,np.nan),"horizon":(t_cible-t_img).days,
            "doy_sin":np.sin(2*np.pi*t_cible.dayofyear/365.25),"doy_cos":np.cos(2*np.pi*t_cible.dayofyear/365.25),
            "temp5":w5.temp_c.mean(),"ray5":w5.ray_mj_m2.sum(),"pluie5":w5.pluie_mm.sum(),
            "debit10":np.log1p(w10.debit.mean()),"flux30":np.log1p(w30.flux.sum()/1000)}
# --- couples d'apprentissage : image i -> image j (3 à 20 jours plus tard)
L=[]
for b,g in c.groupby("baie"):
    g=g.reset_index(drop=True)
    for i in range(len(g)):
        for j in range(i+1,len(g)):
            h=(g.date[j]-g.date[i]).days
            if h<3: continue
            if h>20: break
            L.append({"baie":b,"annee":g.date[j].year,"date0":g.date[i],"date1":g.date[j],"l_ulve1":np.log1p(g.ulve[j]),
                      **variables(b,g.date[i],g.ulve[i],g.date[j])})
d=pd.DataFrame(L).dropna(subset=["niveau0","niveau1"])
BAIES=sorted(c.baie.unique()); VARS=["l_ulve0","niveau0","niveau1","coef1","horizon","doy_sin","doy_cos","temp5","ray5","pluie5","debit10","flux30"]
def X(df):
    x=df[VARS].copy()
    for b in BAIES: x["baie_"+b]=(df.baie==b).astype(float)
    return x
def modeles(tr):
    y=tr.l_ulve1-tr.l_ulve0
    et=make_pipeline(SimpleImputer(strategy="median"),ExtraTreesRegressor(n_estimators=300,min_samples_leaf=5,max_features=0.5,random_state=0)).fit(X(tr),y)
    rg=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),RidgeCV(alphas=np.logspace(-1,4,30))).fit(X(tr),y)
    return lambda df: df.l_ulve0.values+(et.predict(X(df))+rg.predict(X(df)))/2
print("couples d'apprentissage :",len(d))

# --- 1. évaluation : prévision faite 5 jours avant chaque image claire
E=[]; G=[]
for Y in range(2017,2026):
    tr=d[d.annee!=Y]
    if tr.empty: continue
    f=modeles(tr)
    sais=tr.assign(m=tr.date1.dt.month).groupby(["baie","m"]).l_ulve1.median()
    for b,g in c[c.date.dt.year==Y].groupby("baie"):
        allimg=c[c.baie==b]
        for _,r in g.iterrows():
            t=r.date-pd.Timedelta(days=5); base=allimg[(allimg.date<=t)&(allimg.date>=t-pd.Timedelta(days=15))]
            if base.empty: continue
            bi=base.iloc[-1]
            row={"baie":b,"annee":Y,"date1":r.date,"l_ulve1":np.log1p(r.ulve),**variables(b,bi.date,bi.ulve,r.date)}
            if np.isnan(row["niveau1"]) or np.isnan(row["niveau0"]): continue
            row["p_modele"]=f(pd.DataFrame([row]))[0]; row["p_persistance"]=row["l_ulve0"]; row["p_saison"]=sais.get((b,r.date.month),np.nan)
            E.append(row)
    # --- 2. courbe de prévision tous les 5 jours (avril-octobre)
    for b in BAIES:
        allimg=c[c.baie==b]
        for t in pd.date_range("%d-04-01"%Y,"%d-10-31"%Y,freq="5D"):
            base=allimg[(allimg.date<=t)&(allimg.date>=t-pd.Timedelta(days=20))]
            if base.empty: continue
            bi=base.iloc[-1]; cible=t+pd.Timedelta(days=5)
            v=variables(b,bi.date,bi.ulve,cible); vb=variables(b,bi.date,bi.ulve,cible,NIV_BAS[b])
            if np.isnan(v["niveau0"]): continue
            pv=f(pd.DataFrame([{"baie":b,**v}]))[0] if not np.isnan(v["niveau1"]) else np.nan
            pb=f(pd.DataFrame([{"baie":b,**vb}]))[0]
            G.append({"baie":b,"annee":Y,"date_prevision":t,"date_cible":cible,"visible_ha":np.expm1(pv),"basse_mer_ha":np.expm1(pb)})
e=pd.DataFrame(E).dropna(subset=["p_saison"]); gr=pd.DataFrame(G)
res={}
for k in ("p_persistance","p_saison","p_modele"):
    ha=np.expm1(e[k]); v=np.expm1(e.l_ulve1)
    res[k[2:]]={"n":len(e),"erreur moyenne (ha)":(ha-v).abs().mean(),"erreur log":(e[k]-e.l_ulve1).abs().mean(),"à ±50 %":100*((ha>=v/1.5)&(ha<=v*1.5)).mean()}
r=pd.DataFrame(res).T; r["gain vs persistance (%)"]=100*(1-r["erreur log"]/r.loc["persistance","erreur log"])
print("\n=== 1. Prévision à 5 jours de la surface visible (marée PRÉVUE), 2017-2025, une année laissée de côté")
print(r.round(2).to_string())
e.to_csv(S+VERSION+"_sys5j_eval.csv",index=False); gr.to_csv(S+VERSION+"_sys5j_courbes.csv",index=False)

# --- 3. cumul façon CEVA : moyenne mensuelle de la surface « basse mer » prévue, mai + juillet + septembre
gr["mois"]=gr.date_cible.dt.month
cum=gr[gr.mois.isin([5,7,9])].groupby(["baie","annee","mois"]).basse_mer_ha.mean().unstack()
cum=cum.dropna().sum(axis=1).rename("cumul_prevu").reset_index()
t=pd.read_csv("data/processed/baies_annuel.csv"); mo=t.groupby("baie").surface_ha.transform("mean"); t.loc[t.surface_ha<0.05*mo,"surface_ha"]=np.nan
m=cum.merge(t[["baie","annee","surface_ha"]],on=["baie","annee"],how="left")
# facteur d'échelle par baie appris sur les AUTRES années (la détection brute surestime les côtes rocheuses)
m["facteur"]=np.nan
for i,row in m.iterrows():
    o=m[(m.baie==row.baie)&(m.annee!=row.annee)].dropna(subset=["surface_ha"])
    if len(o)>=2: m.loc[i,"facteur"]=o.surface_ha.sum()/o.cumul_prevu.sum()
m["cumul_calibre"]=m.cumul_prevu*m.facteur
m.to_csv(S+VERSION+"_sys5j_cumuls.csv",index=False)
mm=m.dropna(subset=["surface_ha","cumul_calibre"])
def rec(g):
    return pd.Series({"années":len(g),"CEVA moyen":g.surface_ha.mean(),"prévu brut":g.cumul_prevu.mean(),"ratio brut":g.cumul_prevu.sum()/g.surface_ha.sum(),
        "prévu calibré":g.cumul_calibre.mean(),"écart calibré (%)":100*(g.cumul_calibre-g.surface_ha).abs().sum()/g.surface_ha.sum(),
        "suit les années (Spearman)":g.cumul_prevu.corr(g.surface_ha,method="spearman") if len(g)>=3 else np.nan})
R=mm.groupby("baie").apply(rec); R.loc["TOTAL"]=rec(mm)
print("\n=== 3. Cumul mai + juillet + septembre prévu vs CEVA (2017-2022)")
print(R.round(2).to_string())
print("\nDétail (prévu calibré / CEVA, ha) :")
print(mm.assign(txt=lambda x: x.cumul_calibre.round(0).astype(int).astype(str)+" / "+x.surface_ha.round(0).astype(int).astype(str)).pivot(index="baie",columns="annee",values="txt").fillna("—").to_string())
print("\nCumuls prévus 2023-2025 (calibrés, ha) :")
f2=m[m.annee>=2023].copy(); f2["facteur"]=[m[(m.baie==b)].dropna(subset=["surface_ha"]).pipe(lambda o: o.surface_ha.sum()/o.cumul_prevu.sum() if len(o) else np.nan) for b in f2.baie]
print((f2.assign(v=f2.cumul_prevu*f2.facteur).pivot(index="baie",columns="annee",values="v")).round(0).to_string())
