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