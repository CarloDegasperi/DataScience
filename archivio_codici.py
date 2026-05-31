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

'''
# partiamo dalla posizione delle stazioni APPA
appa_pos = appa_pos[['station', 'geometry']]

# strategia 1: assumiamo assenza di moti turbolenti
# creiamo una colonna che sommi la potenza dissipata nella cella della stazione nell'arco di 24 ore
# dato che ogni valore di potenza fa riferimento ad un intervallo di tempo costante (un'ora), sommare la potenza equivale a sommare l'energia consumata

# passiamo dalla notazione standard GPS alla notazione in metri
grid_df_mt = grid_df.set_crs('EPSG:4326').to_crs('EPSG:32632')
appa_pos = gpd.GeoDataFrame(data=appa_pos,geometry='geometry')
appa_pos_mt = appa_pos.set_crs('EPSG:4326').to_crs('EPSG:32632')

radius = 40 # studiamo entro un raggio di 40 metri per le stazioni che sono proprio al limite di una cella rifornita dalla SET
appa_buffer = appa_pos_mt.copy()
appa_buffer[f"geometry_buffer_{radius}"] = appa_buffer.geometry.buffer(radius)
appa_buffer = appa_buffer.set_geometry(f"geometry_buffer_{radius}")
str1_df = gpd.sjoin(appa_buffer,grid_df_mt,predicate="intersects",how="inner")
str1_df = pd.merge(left=str1_df,right=sq_power_df,left_on='cellId',right_on='squareid',how='left')
# sommiamo sulle celle per le stazioni a cui viene assegnata più di una cella
str1_df = str1_df.groupby(['station', 'geometry_x', f'geometry_buffer_{radius}', 'date', 'hour'])['power_square'].sum().reset_index()
str1_df["power_prev_24h"] = str1_df.groupby('station')['power_square'].transform(lambda x: x.shift(1).rolling(window=24, min_periods=1).sum())
str1_df = str1_df[['station', 'date', 'hour', 'power_prev_24h']]
str1_df['hour'] = str1_df['hour'].astype(int)

# strategia 2: i moti turbolenti sono presenti
# aggiungiamo diverse colonne relative al movimento delle particelle nel tempo precedente a quello considerato.
# in questo modo stiamo già parzialmente preparando il dataframe per la classificazione, ma allo stesso tempo stiamo integrando il constraint legato alla 
# mobilità delle particelle di inquinante in aria

K = 1 # coefficiente di diffusione turbolenta - la nostra assunzione è abbastanza approssimativa

final_power_df = fs.get_power_areas(appa_pos_mt, grid_df_mt, sq_power_df, K, 24)
final_power_df

'''


def get_power_areas(appa_pos_mt, grid_df_mt, sq_power_df, K, tot_h):
    dfs = []
    for h in range(tot_h): # voglio l'impatto degli inquinanti entro tot_h
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

    return final_power_df
    final_power_df = final_power_df.dropna(axis=0)

    # sommo sulle diverse aree e mi tengo solo la colonna con l'area totale
    power_cols = [a for a in final_power_df.columns if a.endswith('hb_power_area')]
    final_power_df['tot_area_power'] = final_power_df[power_cols].sum(axis=1)
    final_power_df = final_power_df[['station', 'date', 'hour', 'tot_area_power']]

    return final_power_df