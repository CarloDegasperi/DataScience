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

            # riportiamo innanzitutto tutti i dati rilevati ai quarti d'ora
            for i in range(len(temp_cols)):
                hhmm = temp_cols[i].split(".")[1]
                hour = hhmm[:2]
                minute = hhmm[2:]

                rows.append({
                    'station': station_id,
                    'date': data,
                    'hour': int(hour),
                    'minute': int(minute),
                    'elevation': elevation,
                    'temperature': row[temp_cols[i]],
                    'precipitation': row[prec_cols[i]],
                    'winds_spd': row[winds_cols_spd[i]],
                    'winds_dir': row[winds_cols_dir[i]],
                    'geometry': geometry
                }) 

    organized_df = gpd.GeoDataFrame(rows, geometry="geometry")

    # voglia i dati solo su base oraria, in modo da conformare con i dati APPA.
    # per tenere conto delle informazioni rilevate ogni quarto d'ora, mediamo le informazioni su base oraria.
    # in particolare riferiamo ad una data ora X la media dei dati da (X-1):15 a X:00
    organized_df['datetime'] = pd.to_datetime(organized_df['date']) + pd.to_timedelta(organized_df['hour'], unit='h') + pd.to_timedelta(organized_df['minute'], unit='m')
    organized_df['hour_ref'] = (organized_df['datetime'] + pd.Timedelta(minutes=45)).dt.floor('h')
    organized_df = organized_df.groupby(['station', 'elevation', 'geometry', 'hour_ref'])[['temperature', 'precipitation', 'winds_spd', 'winds_dir']].mean().reset_index()
    organized_df['date'] = organized_df['hour_ref'].dt.date
    organized_df['hour'] = organized_df['hour_ref'].dt.hour

    return organized_df[['station', 'date', 'hour', 'elevation', 'temperature', 'precipitation', 'winds_spd', 'winds_dir', 'geometry']]

########################################################################################################################

def get_AQI_pol(value, intervals):
    if pd.isna(value):
        return np.nan

    for i, (low, high) in enumerate(intervals):
        if low <= value < high:
            return 20 / (high - low) * (value - low) + 20 * i
               
    return np.nan

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

def shift_prec(df, hours=1):
    # con questo sort facciamo in modo che le prime precipitazioni di un giorno vengano shiftate sulle ultime del
    # giorno prima
    return (df.sort_values(['station_appa', 'date', 'hour'])
          .groupby('station_appa')['precipitation']            # raggruppando in base alla stazione evitiamo scambi di dati
          .shift(hours)                                         # effettivo shist
          .sort_index())                                        # le rimettiamo nell'ordine iniziale
