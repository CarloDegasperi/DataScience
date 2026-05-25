import numpy as np
import pandas as pd
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

def get_geometry_APPA(APPA_df, APPA_pos):
    target = {}
    for i in range(len(APPA_pos['data names'])):
        target[APPA_pos['data names'][i]] = Point(APPA_pos['lon'][i], APPA_pos['lat'][i])
    return target

########################################################################################################################

EAQI_THRESHOLDS = {
    'PM10': [(0, 20, 'good'), (21, 35, 'fair'), (36, 50, 'moderate'), (51, 100, 'poor'), (100, np.inf, 'awful')],

    'NO2': [(0, 40, 'good'), (41, 100, 'fair'), (101, 200, 'moderate'), (201, 400, 'poor'), (400, np.inf, 'awful')],

    'O3': [(0, 80, 'good'), (81, 120, 'fair'), (121, 180, 'moderate'), (181, 240, 'poor'), (240, np.inf, 'awful')],

    'SO2': [(0, 100, 'good'), (101, 200, 'fair'), (201, 350, 'moderate'), (351, 500, 'poor'), (500, np.inf, 'awful')],

    'CO': [(0, 5.0, 'good'), (5.1, 7.5, 'fair'), (7.6, 10, 'moderate'), (10.1, 20, 'poor'),(20, np.inf, 'awful')]
}

EAQI_ORDER = {'good': 4, 'fair': 3, 'moderate': 2, 'poor': 1, 'awful': 0}

def classify_pollutant(value, pollutant):

    if pd.isna(value):
        return None

    thresholds = EAQI_THRESHOLDS[pollutant]

    for low, high, category in thresholds:
        if low <= value <= high:
            return category

    return None


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