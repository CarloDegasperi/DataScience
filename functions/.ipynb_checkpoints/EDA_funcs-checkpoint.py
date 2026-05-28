import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def fill_hole(weather):
    new_data = weather[(weather['date'] == '2013-11-08') & (weather['station'] == 'T0408')].copy()

    for c in weather.columns:
        if c in ['station', 'geomPoint.geom', 'elevation', 'date', 'timestamp']:
            new_data[c] = new_data[c]
        else:
            new_data[c] = np.nan
    new_data['date'] = '2013-11-09'
    new_data['timestamp'] = 1383951600

    if((weather[(weather['date'] == '2013-11-09') & (weather['station'] == 'T0408')]).empty):
        weather = pd.concat([weather, new_data], ignore_index=True)

    return weather

########################################################################################################################

def winds_split(weather):
    weather_df = weather

    winds_cols = [c for c in weather_df.columns if c.startswith("winds.")]

    new_cols = []

    for c in winds_cols:

        split_cols = weather_df[c].str.split("@", n=1, expand=True)

        # ensure two columns always exist
        split_cols = split_cols.reindex(columns=[0, 1])

        split_cols.columns = [f"spd_{c}", f"dir_{c}"]

        split_cols = split_cols.apply(pd.to_numeric, errors="coerce")

        new_cols.append(split_cols)

    weather_df = pd.concat([weather_df] + new_cols, axis=1)
    weather_df.drop(columns=winds_cols, inplace=True)
    return weather_df

########################################################################################################################

def restructure_weather(weather_df):
    rows = []

    for station_id in weather_df['station'].drop_duplicates():

        df_station = weather_df[weather_df["station"] == station_id]

        # === COLONNE DATI ===
        temp_cols = [c for c in df_station.columns if c.startswith("temperatures.")]
        prec_cols = [c for c in df_station.columns if c.startswith("precipitations.")]
        winds_cols_spd = [c for c in df_station.columns if (c.startswith("spd_winds."))]
        winds_cols_dir = [c for c in df_station.columns if (c.startswith("dir_winds."))]

        for _, row in df_station.iterrows():
            geometry = row['geometry']
            data = row['date']
            elevation = row['elevation']

            for i in range(len(temp_cols)):
                hhmm = temp_cols[i].split(".")[1]
                hour = hhmm[:2]
                minute = hhmm[2:]

                if(minute == "00"):
                    rows.append({
                        'station': station_id,
                        'date': data,
                        'hour': int(hour),
                        'elevation': elevation,
                        'temperature': row[temp_cols[i]],
                        'precipitation': row[prec_cols[i]],
                        'winds_spd': row[winds_cols_spd[i]],
                        'winds_dir': row[winds_cols_dir[i]],
                        'geometry': geometry
                    }) 

    out_df = gpd.GeoDataFrame(rows, geometry="geometry")
    return out_df

########################################################################################################################

def timed_weather(weather_df):
    t_weather_df = None

    for station_id in weather_df['station'].drop_duplicates():

        df_station = weather_df[weather_df['station'] == station_id]

        # === COLONNE PRECIPITAZIONI ===
        temp_cols = [c for c in df_station.columns if c.startswith('temperatures.')]
        prec_cols = [c for c in df_station.columns if c.startswith('precipitations.')]
        winds_cols_spd = [c for c in df_station.columns if c.startswith('spd_winds.')]
        winds_cols_dir = [c for c in df_station.columns if c.startswith('dir_winds.')]

        # lista finale
        rows = []
        prec = []
        winds_spd = []
        winds_dir = []

        # === COSTRUZIONE NUOVO DATAFRAME ===
        for _, row in df_station.iterrows():

            date = pd.to_datetime(row['date'])

            for i in range(len(temp_cols)):

                # estrae HHMM
                hhmm = temp_cols[i].split('.')[1]

                hour = hhmm[:2]
                minute = hhmm[2:]

                # costruzione datetime
                dt = date.replace(
                    hour=int(hour),
                    minute=int(minute)
                )

                # formato richiesto:
                # minuto-ora-giorno-mese-anno
                # esempio 1215140313
                custom_time = dt.strftime('%M%H%d%m%y')

                rows.append({
                    'datetime': custom_time,
                    'temperatures_' + station_id: row[temp_cols[i]],
                    'precipitations_' + station_id: row[prec_cols[i]],
                    'winds_spd_' + station_id: row[winds_cols_spd[i]],
                    'winds_dir_' + station_id: row[winds_cols_dir[i]]
                }) 
        
        df = pd.DataFrame(rows)

        if t_weather_df is None:
            t_weather_df = df
        else:
            t_weather_df = t_weather_df.merge(df, on='datetime')
    
    return t_weather_df

########################################################################################################################

def get_square_power(pow_pos_df, lines_df):
    el_power_df = pd.merge(pow_pos_df,lines_df,left_on='LINESET',right_on='lineset',how='right')

    # calcolo il totale di ubicazioni per ogni linea
    tot_ub_df = el_power_df.groupby(['lineset', 'date', 'hour'])['NR_UBICAZIONI'].sum().reset_index().drop_duplicates(subset='lineset')
    tot_ub_df = tot_ub_df.rename(columns={'NR_UBICAZIONI': 'tot_ub'})
    el_power_df['tot_ub'] = el_power_df['lineset'].map(tot_ub_df.set_index('lineset')['tot_ub'])

    el_power_df['power_square'] = el_power_df['power'] * el_power_df['NR_UBICAZIONI'] / el_power_df['tot_ub']
    el_power_df = el_power_df.rename(columns={'SQUAREID': 'squareid'})

    # quello che interessa a me è di trovare la potenza media oraria per quandrato. Quindi raggruppo per quadrato e sommo
    sq_power_df = el_power_df.groupby(['squareid', 'date', 'hour'])['power_square'].sum().reset_index()

    return sq_power_df

########################################################################################################################

def get_power_areas(appa_pos_mt, grid_df_mt, sq_power_df, K):
    dfs = []
    for h in range(24): # voglio l'impatto degli inquinanti entro un giorno
        radius = round(np.sqrt(6*K*h*3600) + 1) # distanza in m che assumiamo una particella possa percorrere in h ore
        # sommo 1 per sopperire a h = 0
        appa_buffer = appa_pos_mt
        appa_buffer[f"geometry_buffer_{radius}"] = appa_buffer.geometry.buffer(radius)
        appa_buffer = appa_buffer.set_geometry(f"geometry_buffer_{radius}")
        area_df = gpd.sjoin(appa_buffer,grid_df_mt,predicate="intersects",how="inner")
        area_df = pd.merge(left=area_df,right=sq_power_df,left_on='cellId',right_on='squareid',how='left')
        area_df = area_df.groupby(['station', 'geometry_x', f'geometry_buffer_{radius}', 'date', 'hour']).sum('power_square').reset_index()
        area_df = area_df.rename(columns={'geometry_x': 'geometry', 'power_square': f'power_area_{radius}'})

        area_df = area_df.sort_values(['station', 'date', 'hour'])
        # sto facendo una sorta di integrale sul tempo. Se una particella inquinante arriva ad una data distanza dopo tot ore, shifto la colonna 
        # della potenza di quel numero di ore in modo che l'effetto, come è nella realtà, sia ritardato
        area_df[f'{h}hb_power_area'] = area_df.groupby(['station'])[f'power_area_{radius}'].shift(h)
        area_df = area_df[['station', 'date', 'hour', f'{h}hb_power_area']]

        dfs.append(area_df)

    final_power_df = dfs[0]

    for df in dfs[1:]:
        final_power_df = pd.merge(final_power_df,df,on=['station', 'date', 'hour'],how='outer')

    final_power_df = final_power_df.dropna(axis=0)

    # sommo sulle diverse aree e mi tengo solo la colonna con l'area totale
    power_cols = [a for a in final_power_df.columns if a.endswith('hb_power_area')]
    final_power_df['tot_area_power'] = final_power_df[power_cols].sum(axis=1)
    final_power_df = final_power_df[['station', 'date', 'hour', 'tot_area_power']]

    return final_power_df

########################################################################################################################

EAQI_THRESHOLDS = {
    'PM10': [(0, 20, 'good'), (21, 35, 'fair'), (36, 50, 'moderate'), (51, 100, 'poor'), (100, np.inf, 'awful')],

    'NO2': [(0, 40, 'good'), (41, 100, 'fair'), (101, 200, 'moderate'), (201, 400, 'poor'), (400, np.inf, 'awful')],

    'O3': [(0, 80, 'good'), (81, 120, 'fair'), (121, 180, 'moderate'), (181, 240, 'poor'), (240, np.inf, 'awful')],

    'SO2': [(0, 100, 'good'), (101, 200, 'fair'), (201, 350, 'moderate'), (351, 500, 'poor'), (500, np.inf, 'awful')],

    'CO': [(0, 5.0, 'good'), (5.1, 7.5, 'fair'), (7.6, 10, 'moderate'), (10.1, 20, 'poor'),(20, np.inf, 'awful')]
}

EAQI_ORDER = {'good': 4, 'fair': 3, 'moderate': 2, 'poor': 1, 'awful': 0}

########################################################################################################################

def classify_pollutant(value, pollutant):

    if pd.isna(value):
        return None

    thresholds = EAQI_THRESHOLDS[pollutant]

    for low, high, category in thresholds:
        if low <= value <= high:
            return category

    return None

########################################################################################################################

def add_eaqi(APPA_df):

    APPA_df = APPA_df.copy()

    air_quality = []

    for _, row in APPA_df.iterrows():

        categories = []

        for col in APPA_df.columns:

            if col == 'datetime':
                continue

            value = row[col]

            # Detect pollutant from column name
            pollutant = None

            for p in EAQI_THRESHOLDS.keys():
                if p in col:
                    pollutant = p
                    break

            if pollutant is None:
                continue

            category = classify_pollutant(value, pollutant)

            if category is not None:
                categories.append(category)

        # Worst category determines EAQI
        if len(categories) == 0:
            final_category = None

        else:
            final_category = min(
                categories,
                key=lambda x: EAQI_ORDER[x]
            )

        air_quality.append(final_category)

    APPA_df['EAQI'] = air_quality

    return APPA_df

########################################################################################################################

def SET_time(SET_df):
    time = pd.to_datetime(SET_df['time']).dt
    
    SET_df.drop('time', axis=1)

    SET_df['day'] = time.day - 1
    SET_df['time'] = time.minute + time.hour*60
    SET_df['weekday'] = time.day_of_week

    return SET_df

########################################################################################################################

def get_minute(time_str):
    return int(time_str[0:2])


def get_hour(time_str):
    return int(time_str[2:4])


def get_day(time_str):
    return int(time_str[4:6])


def get_month(time_str):
    return int(time_str[6:8])


def get_year(time_str):
    return 2000 + int(time_str[8:10])


########################################################################################################################

def datetime_to_int(datetime):
    reference = pd.Timestamp('2013-11-01')
    return (pd.Timestamp(datetime) - reference).days

########################################################################################################################

########################################################################################################################

def shift_prec(df, hours=1):
    # con questo sort facciamo in modo che le prime precipitazioni di un giorno vengano shiftate sulle ultime del
    # giorno prima
    return (df.sort_values(['station_appa', 'date', 'hour'])
          .groupby('station_appa')['precipitations']            # raggruppando in base alla stazione evitiamo scambi di dati
          .shift(hours)                                         # effettivo shist
          .sort_index())                                        # le rimettiamo nell'ordine iniziale
