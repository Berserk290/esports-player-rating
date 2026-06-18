import io
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Рейтинг киберспортсменов",
    page_icon="🎮",
    layout="wide"
)


API_URL = "https://api.stratz.com/graphql"

# Токен встроен по твоей просьбе.
DEFAULT_STRATZ_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJTdWJqZWN0IjoiMDI0NWE4NDYtMDkxZi00OWZmLThiNzEtMjBlOTZmYTFmMzhjIiwiU3RlYW1JZCI6IjEyNjI3MDEwNTgiLCJBUElVc2VyIjoidHJ1ZSIsIm5iZiI6MTc3NDUzMTgzNSwiZXhwIjoxODA2MDY3ODM1LCJpYXQiOjE3NzQ1MzE4MzUsImlzcyI6Imh0dHBzOi8vYXBpLnN0cmF0ei5jb20ifQ.q_Jefm9EjgIfGfwaT1spXPOAacO4z5Y_qC9ORxS5tYU"

# ID турнира/лиги из твоего рабочего скрипта.
DEFAULT_LEAGUE_ID = 18324


FEATURES = [
    "avg_kills",
    "avg_deaths",
    "avg_assists",
    "avg_gpm",
    "avg_xpm",
    "winrate"
]

FEATURE_LABELS = {
    "avg_kills": "Kills",
    "avg_deaths": "Deaths",
    "avg_assists": "Assists",
    "avg_gpm": "GPM",
    "avg_xpm": "XPM",
    "winrate": "Winrate"
}

REQUIRED_MATCH_COLUMNS = [
    "match_id",
    "player",
    "team",
    "hero",
    "kills",
    "deaths",
    "assists",
    "gpm",
    "xpm",
    "win"
]


LEAGUE_MATCHES_QUERY = """
query LeagueMatches($leagueId: Int!, $skip: Int!, $take: Int!) {
  league(id: $leagueId) {
    id
    displayName
    matches(request: { skip: $skip, take: $take }) {
      id
    }
  }
}
"""


MATCH_DETAILS_QUERY = """
query MatchDetails($matchId: Long!) {
  match(id: $matchId) {
    id
    startDateTime
    durationSeconds
    didRadiantWin
    radiantTeam { id name }
    direTeam { id name }
    players {
      isRadiant
      steamAccount { id name }
      role
      position
      hero { id displayName }
      kills
      deaths
      assists
      goldPerMinute
      experiencePerMinute
    }
  }
}
"""


def run_stratz_query(token: str, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "STRATZ_API"
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json={"query": query, "variables": variables},
        timeout=40
    )

    response.raise_for_status()
    payload = response.json()

    if "errors" in payload:
        raise RuntimeError(payload["errors"])

    return payload["data"]


def fetch_all_match_ids(token: str, league_id: int, progress_callback=None) -> List[int]:
    match_ids = []
    skip = 0
    take = 100

    while True:
        data = run_stratz_query(
            token=token,
            query=LEAGUE_MATCHES_QUERY,
            variables={
                "leagueId": int(league_id),
                "skip": skip,
                "take": take
            }
        )

        league = data.get("league")
        if not league:
            raise RuntimeError(f"Лига с id={league_id} не найдена или недоступна.")

        matches = league.get("matches") or []

        if not matches:
            break

        match_ids.extend([m["id"] for m in matches if m.get("id") is not None])

        if progress_callback:
            progress_callback(len(match_ids))

        skip += take
        time.sleep(0.2)

    return match_ids


def fetch_match_details(token: str, match_id: int) -> Dict[str, Any]:
    data = run_stratz_query(
        token=token,
        query=MATCH_DETAILS_QUERY,
        variables={"matchId": int(match_id)}
    )

    match = data.get("match")

    if not match:
        raise RuntimeError(f"Матч {match_id} не найден или недоступен.")

    return match


def build_rows_from_match(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    radiant_team = match.get("radiantTeam") or {}
    dire_team = match.get("direTeam") or {}

    radiant_name = radiant_team.get("name") or "Radiant"
    dire_name = dire_team.get("name") or "Dire"

    did_radiant_win = bool(match.get("didRadiantWin"))

    rows = []

    for index, player in enumerate(match.get("players") or [], start=1):
        is_radiant = bool(player.get("isRadiant"))

        team = radiant_name if is_radiant else dire_name
        opponent = dire_name if is_radiant else radiant_name
        player_win = did_radiant_win if is_radiant else not did_radiant_win

        steam_account = player.get("steamAccount") or {}
        hero = player.get("hero") or {}

        player_name = (
            steam_account.get("name")
            or str(steam_account.get("id") or f"Player_{match.get('id')}_{index}")
        )

        rows.append({
            "match_id": match.get("id"),
            "team": team,
            "opponent": opponent,
            "player": player_name,
            "hero": hero.get("displayName") or "unknown",
            "position": player.get("position"),
            "role": player.get("role"),
            "kills": player.get("kills", 0),
            "deaths": player.get("deaths", 0),
            "assists": player.get("assists", 0),
            "gpm": player.get("goldPerMinute", 0),
            "xpm": player.get("experiencePerMinute", 0),
            "win": int(player_win)
        })

    return rows


def fetch_league_dataset(
    token: str,
    league_id: int,
    max_matches: Optional[int] = None
) -> pd.DataFrame:
    status = st.empty()
    progress = st.progress(0)

    def update_match_count(count: int):
        status.info(f"Получаю список матчей турнира. Найдено матчей: {count}")

    match_ids = fetch_all_match_ids(
        token=token,
        league_id=league_id,
        progress_callback=update_match_count
    )

    if max_matches is not None and max_matches > 0:
        match_ids = match_ids[:max_matches]

    if not match_ids:
        raise RuntimeError("По выбранному турниру не найдено матчей.")

    rows = []
    total = len(match_ids)

    for index, match_id in enumerate(match_ids, start=1):
        status.info(f"Обрабатываю матч {index} из {total}: {match_id}")

        match = fetch_match_details(token=token, match_id=match_id)
        rows.extend(build_rows_from_match(match))

        progress.progress(index / total)
        time.sleep(0.2)

    progress.empty()
    status.success(f"Данные получены. Матчей: {total}, строк статистики: {len(rows)}")

    return pd.DataFrame(rows)


def mode_first(series: pd.Series):
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]

    non_null = series.dropna()
    if not non_null.empty:
        return non_null.iloc[0]

    return None


@st.cache_data
def read_input_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return {"data": pd.read_csv(uploaded_file)}

    excel = pd.ExcelFile(uploaded_file)
    sheets = {}

    for sheet in excel.sheet_names:
        sheets[sheet] = pd.read_excel(uploaded_file, sheet_name=sheet)

    return sheets


def detect_dataset(sheets_or_data):
    if "data" in sheets_or_data:
        df = sheets_or_data["data"]
        return "raw", df, None, None

    if "final_rating" in sheets_or_data:
        result = sheets_or_data["final_rating"].copy()
        centers_raw = sheets_or_data.get("cluster_centers_raw")
        centers_z = sheets_or_data.get("cluster_centers_z")
        return "ready", result, centers_raw, centers_z

    first_sheet_name = list(sheets_or_data.keys())[0]
    return "raw", sheets_or_data[first_sheet_name], None, None


def build_player_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_MATCH_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "Во входном наборе данных не найдены обязательные поля: "
            + ", ".join(missing)
        )

    work = df.copy()

    numeric_columns = ["kills", "deaths", "assists", "gpm", "xpm", "win"]

    for col in numeric_columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=numeric_columns)
    work["win"] = work["win"].astype(int)

    player_df = (
        work.groupby("player", as_index=False)
        .agg(
            matches_played=("match_id", "count"),
            main_team=("team", mode_first),
            main_hero=("hero", mode_first),
            avg_kills=("kills", "mean"),
            avg_deaths=("deaths", "mean"),
            avg_assists=("assists", "mean"),
            avg_gpm=("gpm", "mean"),
            avg_xpm=("xpm", "mean"),
            winrate=("win", "mean"),
        )
    )

    return player_df


def prepare_ready_dataset(result: pd.DataFrame) -> pd.DataFrame:
    df = result.copy()

    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f"В готовом рейтинге отсутствует поле: {col}")

    if "matches_played" not in df.columns:
        df["matches_played"] = np.nan

    if "main_team" not in df.columns:
        df["main_team"] = ""

    if "main_hero" not in df.columns:
        df["main_hero"] = ""

    return df


def calculate_player_score(player_df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    z_values = scaler.fit_transform(player_df[FEATURES])

    z_df = pd.DataFrame(
        z_values,
        columns=[f"z_{col}" for col in FEATURES],
        index=player_df.index
    )

    result = pd.concat([player_df.copy(), z_df], axis=1)

    result["rating_score"] = (
        result["z_avg_kills"]
        - result["z_avg_deaths"]
        + result["z_avg_assists"]
        + result["z_avg_gpm"]
        + result["z_avg_xpm"]
        + result["z_winrate"]
    )

    return result


def build_cluster_names(centers_z: pd.DataFrame):
    centers = centers_z.copy()

    if "cluster_score" not in centers.columns:
        centers["cluster_score"] = (
            centers["avg_kills"]
            - centers["avg_deaths"]
            + centers["avg_assists"]
            + centers["avg_gpm"]
            + centers["avg_xpm"]
            + centers["winrate"]
        )

    ordered_clusters = (
        centers
        .sort_values("cluster_score", ascending=False)["cluster"]
        .tolist()
    )

    names = [
        "Высокий рейтинговый профиль",
        "Сильный рейтинговый профиль",
        "Средний рейтинговый профиль",
        "Низкий рейтинговый профиль",
        "Базовый рейтинговый профиль"
    ]

    cluster_names = {}

    for index, cluster_id in enumerate(ordered_clusters):
        if index < len(names):
            cluster_names[cluster_id] = names[index]
        else:
            cluster_names[cluster_id] = f"Рейтинговая группа {index + 1}"

    return cluster_names


def fit_kmeans(player_df: pd.DataFrame, clusters_count: int):
    if len(player_df) < clusters_count:
        raise ValueError(
            f"Количество игроков ({len(player_df)}) меньше числа кластеров ({clusters_count}). "
            f"Уменьшите число кластеров или загрузите больше данных."
        )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(player_df[FEATURES])

    model = KMeans(
        n_clusters=clusters_count,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(x_scaled)

    result = player_df.copy()
    result["cluster"] = labels

    centers_z = pd.DataFrame(
        model.cluster_centers_,
        columns=FEATURES
    )
    centers_z["cluster"] = range(clusters_count)

    centers_z["cluster_score"] = (
        centers_z["avg_kills"]
        - centers_z["avg_deaths"]
        + centers_z["avg_assists"]
        + centers_z["avg_gpm"]
        + centers_z["avg_xpm"]
        + centers_z["winrate"]
    )

    centers_raw = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=FEATURES
    )
    centers_raw["cluster"] = range(clusters_count)
    centers_raw = centers_raw.merge(
        centers_z[["cluster", "cluster_score"]],
        on="cluster",
        how="left"
    )

    result = calculate_player_score(result)

    cluster_names = build_cluster_names(centers_z)
    result["rating_group"] = result["cluster"].map(cluster_names)

    return result, centers_raw, centers_z, x_scaled


def evaluate_k_range(player_df: pd.DataFrame, k_min=2, k_max=7):
    max_allowed_k = min(k_max, len(player_df))

    if max_allowed_k < k_min:
        return pd.DataFrame(columns=["Количество кластеров", "Инерция"])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(player_df[FEATURES])

    rows = []

    for k in range(k_min, max_allowed_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )
        model.fit(x_scaled)

        rows.append({
            "Количество кластеров": k,
            "Инерция": model.inertia_
        })

    return pd.DataFrame(rows)


def format_rating_table(df: pd.DataFrame):
    table = df.copy()

    columns = [
        "rank",
        "player",
        "main_team",
        "matches_played",
        "avg_kills",
        "avg_deaths",
        "avg_assists",
        "avg_gpm",
        "avg_xpm",
        "winrate",
        "rating_score",
        "rating_group"
    ]

    available_columns = [col for col in columns if col in table.columns]
    table = table[available_columns].copy()

    rename_map = {
        "rank": "Место",
        "player": "Игрок",
        "main_team": "Команда",
        "matches_played": "Матчи",
        "avg_kills": "Kills",
        "avg_deaths": "Deaths",
        "avg_assists": "Assists",
        "avg_gpm": "GPM",
        "avg_xpm": "XPM",
        "winrate": "Winrate",
        "rating_score": "Рейтинговый балл",
        "rating_group": "Рейтинговая группа"
    }

    table = table.rename(columns=rename_map)

    numeric_cols = [
        "Kills",
        "Deaths",
        "Assists",
        "GPM",
        "XPM",
        "Winrate",
        "Рейтинговый балл"
    ]

    for col in numeric_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").round(3)

    if "Winrate" in table.columns:
        table["Winrate"] = (table["Winrate"] * 100).round(1).astype(str) + "%"

    return table


def build_cluster_profile_chart(centers_z: pd.DataFrame):
    plot_df = centers_z.copy()

    if "cluster_score" in plot_df.columns:
        plot_df = plot_df.sort_values("cluster_score", ascending=False)

    fig = go.Figure()

    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[FEATURE_LABELS[col] for col in FEATURES],
                y=[row[col] for col in FEATURES],
                mode="lines+markers",
                name=f"Кластер {int(row['cluster'])}"
            )
        )

    fig.update_layout(
        title="Профили кластеров по стандартизованным значениям",
        xaxis_title="Показатель",
        yaxis_title="Z-значение",
        height=520,
        legend_title="Кластеры"
    )

    return fig


def build_correlation_chart(player_df: pd.DataFrame):
    corr = player_df[FEATURES].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        x=[FEATURE_LABELS[col] for col in FEATURES],
        y=[FEATURE_LABELS[col] for col in FEATURES],
        aspect="auto",
        title="Корреляционная матрица признаков"
    )

    fig.update_layout(height=560)
    return fig


def build_elbow_chart(evaluation_df: pd.DataFrame):
    fig = px.line(
        evaluation_df,
        x="Количество кластеров",
        y="Инерция",
        markers=True,
        title="Определение количества кластеров методом локтя"
    )

    fig.update_layout(height=460)
    return fig


def build_variable_dendrogram(player_df: pd.DataFrame):
    scaler = StandardScaler()
    data = pd.DataFrame(
        scaler.fit_transform(player_df[FEATURES]),
        columns=[FEATURE_LABELS[col] for col in FEATURES]
    )

    z = linkage(data.T, method="ward")

    fig, ax = plt.subplots(figsize=(8, 5))

    dendrogram(
        z,
        labels=data.columns.tolist(),
        ax=ax
    )

    ax.set_title("Дендограмма признаков")
    ax.set_xlabel("Признаки")
    ax.set_ylabel("Расстояние")

    plt.tight_layout()

    return fig


def build_player_radar(selected_player: pd.Series):
    categories = [FEATURE_LABELS[col] for col in FEATURES]

    values = [
        selected_player[f"z_{col}"]
        for col in FEATURES
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=str(selected_player["player"])
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        showlegend=False,
        height=520,
        title="Статистический профиль игрока"
    )

    return fig


def make_csv_download(df: pd.DataFrame):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


st.title("🎮 Информационно-аналитическая система «Рейтинг киберспортсменов»")

st.markdown(
    """
    Приложение выполняет автоматизированную обработку статистических данных,
    формирует рейтинговые группы игроков и предоставляет интерактивные
    инструменты для анализа результатов.
    """
)


with st.sidebar:
    st.header("Параметры анализа")

    data_source = st.radio(
        "Источник данных",
        [
            "Получить данные через STRATZ API",
            "Загрузить подготовленный файл"
        ]
    )

    uploaded_file = None

    if data_source == "Загрузить подготовленный файл":
        uploaded_file = st.file_uploader(
            "Загрузите входной набор данных",
            type=["xlsx", "csv"]
        )

    else:
        token = st.text_input(
            "STRATZ API Token",
            value=DEFAULT_STRATZ_TOKEN,
            type="password",
            help="Токен уже встроен в приложение, но при необходимости его можно заменить."
        )

        league_id = st.number_input(
            "ID турнира / лиги STRATZ",
            min_value=1,
            value=DEFAULT_LEAGUE_ID,
            step=1
        )

        limit_matches = st.checkbox(
            "Ограничить число матчей для тестового запуска",
            value=True
        )

        max_matches = None

        if limit_matches:
            max_matches = st.number_input(
                "Максимум матчей",
                min_value=1,
                value=20,
                step=1
            )

        fetch_button = st.button(
            "Получить данные из STRATZ",
            type="primary"
        )

        if fetch_button:
            if not token.strip():
                st.error("Введите STRATZ API Token.")
            else:
                try:
                    api_df = fetch_league_dataset(
                        token=token.strip(),
                        league_id=int(league_id),
                        max_matches=int(max_matches) if max_matches else None
                    )

                    st.session_state["api_df"] = api_df
                    st.session_state["api_league_id"] = int(league_id)

                    st.success(
                        f"Данные загружены: {len(api_df)} строк статистики."
                    )

                except Exception as error:
                    st.error("Не удалось получить данные из STRATZ API.")
                    st.exception(error)

    clusters_count = st.slider(
        "Количество кластеров",
        min_value=2,
        max_value=7,
        value=4,
        step=1
    )

    st.divider()

    st.caption(
        "Система поддерживает получение данных через API и загрузку заранее подготовленного набора."
    )


if data_source == "Загрузить подготовленный файл" and uploaded_file is None:
    st.info("Загрузите файл с данными, чтобы выполнить анализ.")
    st.stop()

if data_source == "Получить данные через STRATZ API" and "api_df" not in st.session_state:
    st.info(
        "Нажмите «Получить данные из STRATZ». "
        "Для быстрой проверки можно оставить ограничение по числу матчей."
    )
    st.stop()


try:
    if data_source == "Получить данные через STRATZ API":
        detected_df = st.session_state["api_df"]
        dataset_type = "raw"
        ready_centers_raw = None
        ready_centers_z = None

        with st.expander("Просмотр полученных исходных данных"):
            st.dataframe(
                detected_df.head(50),
                use_container_width=True
            )

    else:
        sheets_or_data = read_input_file(uploaded_file)
        dataset_type, detected_df, ready_centers_raw, ready_centers_z = detect_dataset(sheets_or_data)

    if dataset_type == "raw":
        player_dataset = build_player_dataset(detected_df)
        result, centers_raw, centers_z, x_scaled = fit_kmeans(player_dataset, clusters_count)

    else:
        player_dataset = prepare_ready_dataset(detected_df)

        if "cluster" not in player_dataset.columns:
            result, centers_raw, centers_z, x_scaled = fit_kmeans(player_dataset, clusters_count)

        else:
            result = calculate_player_score(player_dataset)

            if ready_centers_z is not None and "cluster_score" in ready_centers_z.columns:
                cluster_names = build_cluster_names(ready_centers_z)
            else:
                result, centers_raw, centers_z, x_scaled = fit_kmeans(
                    player_dataset,
                    clusters_count
                )
                cluster_names = None

            if "rating_group" not in result.columns:
                if cluster_names is not None:
                    result["rating_group"] = result["cluster"].map(cluster_names)
                else:
                    result["rating_group"] = result["cluster"].astype(str)

            if ready_centers_raw is not None:
                centers_raw = ready_centers_raw.copy()
            else:
                centers_raw = None

            if ready_centers_z is not None:
                centers_z = ready_centers_z.copy()
            else:
                _, centers_raw, centers_z, x_scaled = fit_kmeans(
                    player_dataset,
                    clusters_count
                )

    result = result.sort_values("rating_score", ascending=False).reset_index(drop=True)

    if "rank" in result.columns:
        result = result.drop(columns=["rank"])

    result.insert(0, "rank", result.index + 1)

except Exception as error:
    st.error("Не удалось обработать данные.")
    st.exception(error)
    st.stop()


tab_dashboard, tab_rating, tab_player, tab_clusters, tab_features, tab_export = st.tabs(
    [
        "Обзор",
        "Рейтинг",
        "Карточка игрока",
        "Кластеры",
        "Признаки",
        "Экспорт"
    ]
)


with tab_dashboard:
    st.subheader("Общая информация")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Игроков", len(result))

    if "matches_played" in result.columns:
        col2.metric("Игровых наблюдений", int(result["matches_played"].sum()))
    else:
        col2.metric("Игровых наблюдений", "—")

    col3.metric("Кластеров", result["cluster"].nunique())

    if "winrate" in result.columns:
        col4.metric("Средний Winrate", f"{result['winrate'].mean() * 100:.1f}%")
    else:
        col4.metric("Средний Winrate", "—")

    st.divider()

    left, right = st.columns([1.1, 1])

    with left:
        cluster_counts = (
            result["rating_group"]
            .value_counts()
            .reset_index()
        )

        cluster_counts.columns = ["Рейтинговая группа", "Количество игроков"]

        fig = px.bar(
            cluster_counts,
            x="Рейтинговая группа",
            y="Количество игроков",
            title="Распределение игроков по рейтинговым группам"
        )

        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_players = format_rating_table(result.head(10))

        st.markdown("#### Топ-10 игроков по рейтинговому баллу")
        st.dataframe(
            top_players,
            use_container_width=True,
            hide_index=True
        )


with tab_rating:
    st.subheader("Итоговый рейтинг игроков")

    group_filter = st.multiselect(
        "Фильтр по рейтинговой группе",
        options=sorted(result["rating_group"].dropna().unique().tolist()),
        default=sorted(result["rating_group"].dropna().unique().tolist())
    )

    filtered = result[result["rating_group"].isin(group_filter)].copy()

    st.dataframe(
        format_rating_table(filtered),
        use_container_width=True,
        hide_index=True
    )


with tab_player:
    st.subheader("Индивидуальная карточка игрока")

    players = result["player"].astype(str).tolist()

    selected_name = st.selectbox(
        "Выберите игрока",
        options=players
    )

    selected_player = result[result["player"].astype(str) == selected_name].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Место в рейтинге", int(selected_player["rank"]))
    col2.metric("Группа", selected_player["rating_group"])
    col3.metric("Рейтинговый балл", f"{selected_player['rating_score']:.2f}")

    if "winrate" in selected_player:
        col4.metric("Winrate", f"{selected_player['winrate'] * 100:.1f}%")
    else:
        col4.metric("Winrate", "—")

    st.divider()

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("#### Основные показатели")

        player_info = {
            "Игрок": selected_player.get("player", ""),
            "Команда": selected_player.get("main_team", ""),
            "Основной герой": selected_player.get("main_hero", ""),
            "Количество матчей": selected_player.get("matches_played", ""),
            "Kills": round(float(selected_player.get("avg_kills", 0)), 3),
            "Deaths": round(float(selected_player.get("avg_deaths", 0)), 3),
            "Assists": round(float(selected_player.get("avg_assists", 0)), 3),
            "GPM": round(float(selected_player.get("avg_gpm", 0)), 3),
            "XPM": round(float(selected_player.get("avg_xpm", 0)), 3),
            "Winrate": f"{float(selected_player.get('winrate', 0)) * 100:.1f}%"
        }

        st.table(
            pd.DataFrame(
                player_info.items(),
                columns=["Показатель", "Значение"]
            )
        )

    with right:
        st.plotly_chart(
            build_player_radar(selected_player),
            use_container_width=True
        )

with tab_clusters:
    st.subheader("Анализ кластеров")

    left, right = st.columns([1, 1])

    with left:
        cluster_counts = (
            result
            .groupby(["cluster", "rating_group"])
            .size()
            .reset_index(name="Количество игроков")
        )

        fig = px.bar(
            cluster_counts,
            x="rating_group",
            y="Количество игроков",
            color="cluster",
            title="Количество игроков в рейтинговых группах",
            labels={
                "rating_group": "Рейтинговая группа",
                "cluster": "Кластер"
            }
        )

        fig.update_layout(height=480)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:
        st.plotly_chart(
            build_cluster_profile_chart(centers_z),
            use_container_width=True
        )

    st.markdown("#### Центры кластеров")

    if centers_raw is not None:
        centers_table = centers_raw.copy()

        for col in FEATURES:
            if col in centers_table.columns:
                centers_table[col] = pd.to_numeric(
                    centers_table[col],
                    errors="coerce"
                ).round(3)

        st.dataframe(
            centers_table,
            use_container_width=True,
            hide_index=True
        )


with tab_features:
    st.subheader("Анализ признаков")

    st.plotly_chart(
        build_correlation_chart(player_dataset),
        use_container_width=True
    )

    st.divider()

    evaluation = evaluate_k_range(player_dataset, 2, 7)

    if not evaluation.empty:
        st.plotly_chart(
            build_elbow_chart(evaluation),
            use_container_width=True
        )
    else:
        st.warning("Недостаточно игроков для построения графика метода локтя.")

    st.divider()

    st.markdown("#### Дендограмма признаков")
    st.pyplot(build_variable_dendrogram(player_dataset))


with tab_export:
    st.subheader("Экспорт результатов")

    st.markdown(
        """
        Сформированный рейтинг может быть сохранён для последующего анализа,
        подготовки отчётов или включения результатов в выпускную квалификационную работу.
        """
    )

    export_table = format_rating_table(result)

    st.download_button(
        label="Скачать итоговый рейтинг",
        data=make_csv_download(export_table),
        file_name="player_rating_result.csv",
        mime="text/csv"
    )

    if data_source == "Получить данные через STRATZ API":
        st.download_button(
            label="Скачать полученные исходные данные",
            data=make_csv_download(detected_df),
            file_name="stratz_raw_dataset.csv",
            mime="text/csv"
        )

    st.markdown("#### Предпросмотр экспортируемых данных")

    st.dataframe(
        export_table,
        use_container_width=True,
        hide_index=True
    )