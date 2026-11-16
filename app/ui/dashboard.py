import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import time
import cv2
import numpy as np
import plotly.graph_objects as go
import torch
import streamlit as st

from app.config import DEFAULT_MAP_CENTER, DEFAULT_MAP_ZOOM, EMPLOYEES_DIR
from app.database.db import (
    init_db,
    get_cameras,
    add_camera,
    delete_camera,
    get_all_employees,
    add_employee,
    delete_employee,
    get_recent_incidents,
    set_incident_acknowledged,
    acknowledge_all_incidents,
    delete_incident,
    clear_all_incidents,
    update_camera_settings,
    load_perimeter_polygon,
    save_perimeter_polygon,
    DEFAULT_PERIMETER_POINTS,
)
from app.cv_module.detector import PerimeterDetector, UniversalVideoStream

st.set_page_config(
    page_title="PerimeterGuard AI — Интеллектуальный мониторинг периметра",
    page_icon="🛡️",
    layout="wide",
)

init_db()

# Инициализация состояния для кликов по карте и навигации ревью
if "picked_lat" not in st.session_state:
    st.session_state["picked_lat"] = DEFAULT_MAP_CENTER[0]
if "picked_lon" not in st.session_state:
    st.session_state["picked_lon"] = DEFAULT_MAP_CENTER[1]
if "last_click_sig" not in st.session_state:
    st.session_state["last_click_sig"] = None


@st.cache_resource
def load_detector() -> PerimeterDetector:
    return PerimeterDetector(camera_id=1, line_orientation="vertical", line_position=0.5)


detector = load_detector()


def build_plotly_perimeter_map(
    cameras,
    recent_incidents,
    polygon_points,
    active_alarm_cam_id=None,
    enable_click_grid=False,
    picked_point=None,
    map_style="open-street-map",
    zoom=DEFAULT_MAP_ZOOM,
):
    """
    Строит интерактивную ГИС-карту на базе Plotly MapLibre GL.
    Поддерживает произвольный многоугольник периметра, маркеры камер и точный выбор координат кликом.
    """
    fig = go.Figure()

    if polygon_points:
        center_lat = float(np.mean([p[0] for p in polygon_points]))
        center_lon = float(np.mean([p[1] for p in polygon_points]))
    elif picked_point is not None:
        center_lat, center_lon = picked_point
    elif cameras:
        center_lat = cameras[0].latitude
        center_lon = cameras[0].longitude
    else:
        center_lat, center_lon = DEFAULT_MAP_CENTER

    # 1. Плотная невидимая сетка (150x150, шаг ~4 м) для выбора практически любой точки кликом
    if enable_click_grid:
        lats = np.linspace(center_lat - 0.0042, center_lat + 0.0042, 150)
        lons = np.linspace(center_lon - 0.0062, center_lon + 0.0062, 150)
        grid_lat, grid_lon = np.meshgrid(lats, lons)
        flat_lat = np.round(grid_lat.ravel(), 6)
        flat_lon = np.round(grid_lon.ravel(), 6)
        hover_texts = [
            f"📍 {la:.6f}, {lo:.6f}<br>Кликните для выбора точки"
            for la, lo in zip(flat_lat, flat_lon)
        ]
        fig.add_trace(
            go.Scattermap(
                lat=flat_lat,
                lon=flat_lon,
                mode="markers",
                marker=dict(size=9, opacity=0),
                text=hover_texts,
                hoverinfo="text",
                name="Координатная сетка",
                showlegend=False,
            )
        )

    # 2. Отрисовка контура охраняемого периметра (многоугольник любой формы)
    if len(polygon_points) >= 1:
        poly_lats = [p[0] for p in polygon_points]
        poly_lons = [p[1] for p in polygon_points]

        if len(polygon_points) >= 3:
            closed_lats = poly_lats + [poly_lats[0]]
            closed_lons = poly_lons + [poly_lons[0]]
            fig.add_trace(
                go.Scattermap(
                    lat=closed_lats,
                    lon=closed_lons,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(37, 99, 235, 0.16)",
                    line=dict(color="#2563eb", width=3),
                    hoverinfo="skip",
                    name="Контур периметра",
                )
            )
        elif len(polygon_points) == 2:
            fig.add_trace(
                go.Scattermap(
                    lat=poly_lats,
                    lon=poly_lons,
                    mode="lines",
                    line=dict(color="#2563eb", width=3),
                    hoverinfo="skip",
                    name="Линия периметра",
                )
            )

        vertex_labels = [f"#{i + 1}" for i in range(len(polygon_points))]
        vertex_hovers = [
            f"Узел периметра #{i + 1}<br>Широта: {p[0]:.6f}<br>Долгота: {p[1]:.6f}"
            for i, p in enumerate(polygon_points)
        ]
        fig.add_trace(
            go.Scattermap(
                lat=poly_lats,
                lon=poly_lons,
                mode="markers+text",
                marker=dict(size=11, color="#1d4ed8"),
                text=vertex_labels,
                textposition="top center",
                textfont=dict(size=12, color="#1e3a8a"),
                hovertext=vertex_hovers,
                hoverinfo="text",
                name="Вершины периметра",
            )
        )

    # 3. Маркер выбранной точки (для добавления камеры или навигации)
    if picked_point is not None:
        fig.add_trace(
            go.Scattermap(
                lat=[picked_point[0]],
                lon=[picked_point[1]],
                mode="markers+text",
                marker=dict(size=15, color="#9333ea"),
                text=["🎯 Выбранная точка"],
                textposition="bottom center",
                textfont=dict(size=12, color="#581c87"),
                hovertext=[f"Выбранная точка: {picked_point[0]:.6f}, {picked_point[1]:.6f}"],
                hoverinfo="text",
                name="Выбранная точка",
            )
        )

    # 4. Отрисовка рубежей видеонаблюдения (камер)
    active_by_cam = {}
    for inc in recent_incidents:
        if not inc.is_acknowledged and inc.alert_priority in ("CRITICAL", "HIGH"):
            if inc.camera_id not in active_by_cam:
                active_by_cam[inc.camera_id] = inc

    if cameras:
        cam_lats, cam_lons, cam_colors, cam_sizes, cam_texts, cam_hovers = [], [], [], [], [], []
        alarm_ring_lats, alarm_ring_lons = [], []

        for cam in cameras:
            alarm_inc = active_by_cam.get(cam.id)
            is_alarm = (active_alarm_cam_id == cam.id) or (alarm_inc is not None)

            cam_lats.append(cam.latitude)
            cam_lons.append(cam.longitude)
            cam_texts.append(f"📹 {cam.name}")

            if is_alarm:
                prio = alarm_inc.alert_priority if alarm_inc else "HIGH"
                cam_colors.append("#dc2626" if prio == "CRITICAL" else "#ea580c")
                cam_sizes.append(18)
                alarm_ring_lats.append(cam.latitude)
                alarm_ring_lons.append(cam.longitude)
                hover = (
                    f"<b>🚨 ТРЕВОГА ({prio}): {cam.name}</b><br>"
                    f"Сектор: {cam.sector_description or '—'}<br>"
                    f"Статус: {alarm_inc.person_status if alarm_inc else 'Нарушитель'}<br>"
                    f"Маска: {'Да' if (alarm_inc and alarm_inc.has_mask) else 'Нет'}<br>"
                    f"Координаты: {cam.latitude:.5f}, {cam.longitude:.5f}"
                )
            else:
                cam_colors.append("#16a34a")
                cam_sizes.append(15)
                hover = (
                    f"<b>✅ {cam.name}</b><br>"
                    f"Сектор: {cam.sector_description or '—'}<br>"
                    f"Протокол: {cam.protocol_type}<br>"
                    f"Статус: Штатный режим<br>"
                    f"Координаты: {cam.latitude:.5f}, {cam.longitude:.5f}"
                )
            cam_hovers.append(hover)

        if alarm_ring_lats:
            fig.add_trace(
                go.Scattermap(
                    lat=alarm_ring_lats,
                    lon=alarm_ring_lons,
                    mode="markers",
                    marker=dict(size=38, color="rgba(220, 38, 38, 0.28)"),
                    hoverinfo="skip",
                    name="Зона тревоги",
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scattermap(
                lat=cam_lats,
                lon=cam_lons,
                mode="markers+text",
                marker=dict(size=cam_sizes, color=cam_colors),
                text=cam_texts,
                textposition="bottom center",
                textfont=dict(size=12, color="#111827"),
                hovertext=cam_hovers,
                hoverinfo="text",
                name="Камеры периметра",
            )
        )

    fig.update_layout(
        map=dict(
            style=map_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=430,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            font=dict(color="#111827", size=12),
        ),
        clickmode="event+select",
    )
    return fig


# --- ВЕРХНЯЯ ПАНЕЛЬ СТАТУСА ---
st.title("🛡️ PerimeterGuard AI — Центр видеоаналитики и контроля периметра")

cameras_list = get_cameras()
employees_list = get_all_employees()
incidents_list = get_recent_incidents(limit=200)
polygon_points = load_perimeter_polygon()

unack_incidents = [i for i in incidents_list if not i.is_acknowledged]
critical_count = sum(1 for i in incidents_list if i.alert_priority == "CRITICAL")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("📹 Активных рубежей (камер)", len(cameras_list))
kpi2.metric("🪪 База доступа («Свой»)", f"{len(employees_list)} чел.")
kpi3.metric("🚨 Ожидают ревью", len(unack_incidents), delta=f"Всего в архиве: {len(incidents_list)}", delta_color="off")
kpi4.metric("😷 Сокрытие лица / маска", critical_count)

# --- БОКОВАЯ ПАНЕЛЬ ПОДКЛЮЧЕНИЯ К ПОТОКУ ---
with st.sidebar:
    st.header("📡 Подключение видеопотока")

    if not cameras_list:
        st.warning("Добавьте хотя бы одну камеру во вкладке «Конфигуратор карты и камер».")
        st.stop()

    cam_options = {f"[{c.id}] {c.name}": c for c in cameras_list}
    selected_cam_label = st.selectbox("Выбор рубежа охраны:", list(cam_options.keys()))
    selected_cam = cam_options[selected_cam_label]

    st.caption(f"Протокол: **{selected_cam.protocol_type}** | {selected_cam.sector_description or ''}")

    video_src_input = st.text_input(
        "Адрес потока (RTSP / HTTP / Индекс):",
        value=selected_cam.video_source,
        help="Примеры: rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101 | http://192.168.0.160:8080/video | 0",
    )

    with st.expander("🔌 Конструктор RTSP-ссылки (IP-камеры)"):
        rtsp_vendor = st.selectbox("Производитель / Профиль:", ["Hikvision / HiWatch", "Dahua / Imou", "Generic ONVIF"])
        rtsp_ip = st.text_input("IP-адрес камеры:", value="192.168.1.64")
        rtsp_user = st.text_input("Логин:", value="admin")
        rtsp_pass = st.text_input("Пароль:", value="12345", type="password")
        if rtsp_vendor == "Hikvision / HiWatch":
            built_rtsp = f"rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_ip}:554/Streaming/Channels/101"
        elif rtsp_vendor == "Dahua / Imou":
            built_rtsp = f"rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_ip}:554/cam/realmonitor?channel=1&subtype=0"
        else:
            built_rtsp = f"rtsp://{rtsp_user}:{rtsp_pass}@{rtsp_ip}:554/stream1"
        st.code(built_rtsp, language="text")
        if st.button("Применить RTSP к текущему рубежу", use_container_width=True):
            update_camera_settings(selected_cam.id, built_rtsp, "RTSP")
            st.rerun()

    st.divider()
    st.subheader("📐 Виртуальный рубеж в кадре")
    line_orient = st.radio(
        "Ориентация контрольной линии:",
        options=["vertical", "horizontal"],
        format_func=lambda x: "Вертикальная (проход слева/справа)" if x == "vertical" else "Горизонтальная (пересечение сверху/снизу)",
    )
    line_pos = st.slider("Смещение линии (%):", 15, 85, 50) / 100.0
    mirror_frame = st.checkbox("Зеркальное отображение кадра", value=False)

    map_theme = st.selectbox(
        "Подложка ГИС-карты:",
        options=["open-street-map", "carto-positron", "carto-darkmatter"],
        format_func=lambda x: {
            "open-street-map": "OpenStreetMap (Детальная)",
            "carto-positron": "Carto Light (Светлая)",
            "carto-darkmatter": "Carto Dark (Ночная)",
        }[x],
    )

    detector.camera_id = selected_cam.id
    detector.line_orientation = line_orient
    detector.line_position = line_pos

    st.divider()
    st.subheader("⚡ Быстрая регистрация пропуска")
    quick_name = st.text_input("ФИО сотрудника:", value="Чернецов Максим Евгеньевич")
    quick_pos = st.text_input("Должность:", value="Инженер безопасности")
    register_btn_placeholder = st.empty()

    st.divider()
    col_sb1, col_sb2 = st.columns(2)
    if col_sb1.button("✅ Снять все тревоги", use_container_width=True):
        acknowledge_all_incidents()
        st.rerun()
    if col_sb2.button("🗑️ Очистить архив", use_container_width=True):
        clear_all_incidents()
        detector.last_alert_info = None
        st.rerun()

# --- ОСНОВНОЙ ЭКРАН ОПЕРАТОРА ---
col_video, col_map = st.columns([1.15, 1.0])

with col_video:
    st.subheader(f"🎥 Видеоаналитика: {selected_cam.name}")
    ctrl1, ctrl2 = st.columns(2)
    start_stream = ctrl1.button("▶️ Подключить поток", type="primary", use_container_width=True)
    stop_stream = ctrl2.button("⏹️ Отключить", use_container_width=True)

    video_placeholder = st.empty()
    live_alert_placeholder = st.empty()

with col_map:
    st.subheader("🗺️ Оперативная ГИС-карта объекта")
    map_placeholder = st.empty()
    review_panel_placeholder = st.empty()


def render_map_and_review_panel(active_cam_id=None, chart_key_suffix="init"):
    fresh_incidents = get_recent_incidents(limit=200)
    fresh_cams = get_cameras()
    fresh_poly = load_perimeter_polygon()

    fig = build_plotly_perimeter_map(
        fresh_cams,
        fresh_incidents,
        fresh_poly,
        active_alarm_cam_id=active_cam_id,
        enable_click_grid=False,
        map_style=map_theme,
    )
    with map_placeholder.container():
        st.plotly_chart(fig, use_container_width=True, key=f"op_map_{chart_key_suffix}")

    with review_panel_placeholder.container():
        if fresh_incidents:
            # Выбор ЛЮБОГО инцидента для оперативного ревью прямо под картой
            inc_labels = {}
            for inc in fresh_incidents:
                status_mark = "✅" if inc.is_acknowledged else "🚨"
                inc_labels[
                    f"{status_mark} Инцидент #{inc.id} | {inc.timestamp.strftime('%H:%M:%S')} | {inc.alert_priority} ({inc.person_status})"
                ] = inc

            selected_inc_label = st.selectbox(
                "🔍 Оперативное ревью инцидента (выберите любой из списка):",
                options=list(inc_labels.keys()),
                key=f"review_select_{chart_key_suffix}",
            )
            target_inc = inc_labels[selected_inc_label]
            cam_obj = target_inc.camera

            pr_icon = (
                "🔴 КРИТИЧЕСКИЙ"
                if target_inc.alert_priority == "CRITICAL"
                else ("🟠 ВЫСОКИЙ" if target_inc.alert_priority == "HIGH" else "🟢 ШТАТНЫЙ ДОСТУП")
            )
            ack_badge = "✅ Отработано" if target_inc.is_acknowledged else "⏳ Требует ревью"

            st.markdown(
                f"**Приоритет:** {pr_icon} | **Статус ревью:** {ack_badge} | **Рубеж:** {cam_obj.name}"
            )
            c_img1, c_img2, c_info = st.columns([1.2, 0.8, 1.5])
            full_p = BASE_DIR / target_inc.full_frame_path
            if full_p.exists():
                c_img1.image(str(full_p), caption="Кадр фиксации", use_container_width=True)
            if target_inc.face_crop_path and (BASE_DIR / target_inc.face_crop_path).exists():
                c_img2.image(str(BASE_DIR / target_inc.face_crop_path), caption="Фрагмент лица", use_container_width=True)
            with c_info:
                emp_txt = target_inc.employee.full_name if target_inc.employee else "Не идентифицирован"
                st.write(f"**Субъект:** {emp_txt} (`{target_inc.person_status}`)")
                st.write(f"**Маска / сокрытие:** {'⚠️ ДА' if target_inc.has_mask else 'Нет'}")
                st.write(f"**Сходство вектора:** `{target_inc.similarity_score:.3f}`")

                b1, b2 = st.columns(2)
                if not target_inc.is_acknowledged:
                    if b1.button("✅ Подтвердить", key=f"ack_op_{target_inc.id}_{chart_key_suffix}", use_container_width=True):
                        set_incident_acknowledged(target_inc.id, True)
                        st.rerun()
                else:
                    if b1.button("↩️ В тревоги", key=f"unack_op_{target_inc.id}_{chart_key_suffix}", use_container_width=True):
                        set_incident_acknowledged(target_inc.id, False)
                        st.rerun()
                if b2.button("🗑️ Удалить", key=f"del_op_{target_inc.id}_{chart_key_suffix}", use_container_width=True):
                    delete_incident(target_inc.id)
                    st.rerun()
        else:
            st.info("Периметр под контролем. Зафиксированных инцидентов нет.")


render_map_and_review_panel(chart_key_suffix="static")

# --- ЦИКЛ ПОТОКОВОЙ ОБРАБОТКИ ---
if start_stream:
    update_camera_settings(selected_cam.id, video_src_input)
    cap = UniversalVideoStream(video_src_input)

    if not cap.isOpened():
        live_alert_placeholder.error(
            f"❌ Ошибка подключения к `{video_src_input}`. Проверьте сетевую доступность камеры и учетные данные."
        )
    else:
        device_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        live_alert_placeholder.success(f"✅ Поток активен | Аппаратное ускорение: {device_label}")
        reg_clicked = register_btn_placeholder.button(
            "📸 Зарегистрировать лицо из кадра", use_container_width=True
        )
        event_counter = 0

        while cap.isOpened() and not stop_stream:
            ret, frame = cap.read()
            if not ret:
                live_alert_placeholder.warning("Видеопоток прерван или завершен.")
                break

            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * (1280 / w))))

            if mirror_frame:
                frame = cv2.flip(frame, 1)

            if reg_clicked:
                emb, _ = detector.biometrics.extract_embedding(frame)
                if emb is not None:
                    safe_filename = f"emp_{int(time.time())}.jpg"
                    save_path = EMPLOYEES_DIR / safe_filename
                    cv2.imwrite(str(save_path), frame)
                    add_employee(
                        full_name=quick_name,
                        position=quick_pos,
                        photo_path=f"data/employees/{safe_filename}",
                        embedding=emb,
                        has_access=True,
                    )
                    detector.biometrics.reload_employees()
                    st.toast(f"✅ Субъект {quick_name} внесен в базу доступа!", icon="🪪")
                else:
                    st.toast("⚠️ Лицо не распознано. Убедитесь, что лицо открыто и освещено.", icon="⚠️")
                reg_clicked = False

            annotated_bgr, incident_event = detector.process_frame(frame)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            if incident_event is not None:
                event_counter += 1
                st.toast(
                    f"🚨 ТРЕВОГА [{incident_event['priority']}]: {incident_event['name']}",
                    icon="🚨",
                )
                render_map_and_review_panel(
                    active_cam_id=selected_cam.id,
                    chart_key_suffix=f"live_{event_counter}",
                )

        cap.release()

st.divider()

# --- НИЖНИЕ МОДУЛИ СИСТЕМЫ ---
tab_map_editor, tab_log, tab_emp, tab_analytics = st.tabs([
    "🗺️ Конфигуратор карты и периметра",
    "🚨 Полный журнал и ревью инцидентов",
    "👥 База доступа («Свой — Чужой»)",
    "📊 Аналитика и диагностика ИИ-ядра",
])

with tab_map_editor:
    st.subheader("Интерактивное развертывание рубежей и геометрии периметра")

    click_action = st.radio(
        "Действие при клике мышью по карте:",
        options=["add_vertex", "pick_camera"],
        format_func=lambda x: (
            "⬡ Сразу добавлять вершину в контур периметра"
            if x == "add_vertex"
            else "📍 Выбрать координаты для новой камеры видеонаблюдения"
        ),
        horizontal=True,
    )

    ed_col1, ed_col2 = st.columns([1.25, 1.0])

    with ed_col1:
        editor_fig = build_plotly_perimeter_map(
            cameras_list,
            incidents_list,
            polygon_points,
            enable_click_grid=True,
            picked_point=(st.session_state["picked_lat"], st.session_state["picked_lon"]),
            map_style=map_theme,
        )
        selection_event = st.plotly_chart(
            editor_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=["points"],
            key="interactive_editor_map",
        )

        # Обработка клика пользователя по карте
        if selection_event and selection_event.get("selection") and selection_event["selection"].get("points"):
            pt = selection_event["selection"]["points"][0]
            if "lat" in pt and "lon" in pt:
                new_lat = round(float(pt["lat"]), 6)
                new_lon = round(float(pt["lon"]), 6)
                click_sig = f"{click_action}_{new_lat}_{new_lon}_{len(polygon_points)}"
                if st.session_state["last_click_sig"] != click_sig:
                    st.session_state["last_click_sig"] = click_sig
                    st.session_state["picked_lat"] = new_lat
                    st.session_state["picked_lon"] = new_lon
                    if click_action == "add_vertex":
                        updated_poly = polygon_points + [[new_lat, new_lon]]
                        save_perimeter_polygon(updated_poly)
                        st.rerun()

        st.caption(
            f"🎯 Последняя выбранная точка: **{st.session_state['picked_lat']:.6f}, {st.session_state['picked_lon']:.6f}**"
        )

    with ed_col2:
        # 1. Управление контуром периметра (кликом или ручным вводом координат)
        with st.expander("⬡ Редактор многоугольника периметра", expanded=True):
            p_ctrl1, p_ctrl2, p_ctrl3 = st.columns(3)
            if p_ctrl1.button("🆕 Новый периметр", help="Удалить все точки и нарисовать заново", use_container_width=True):
                save_perimeter_polygon([])
                st.session_state["last_click_sig"] = None
                st.rerun()
            if p_ctrl2.button("↩️ Удалить последнюю", disabled=len(polygon_points) == 0, use_container_width=True):
                save_perimeter_polygon(polygon_points[:-1])
                st.session_state["last_click_sig"] = None
                st.rerun()
            if p_ctrl3.button("🔄 Сбросить контур", use_container_width=True):
                save_perimeter_polygon(DEFAULT_PERIMETER_POINTS)
                st.session_state["last_click_sig"] = None
                st.rerun()

            # Ручной ввод координат для добавления новой вершины или перехода к точке на карте
            st.markdown("**Добавить узел периметра вручную (по координатам):**")
            mc1, mc2, mc3 = st.columns([1.0, 1.0, 1.0])
            manual_lat = mc1.number_input(
                "Широта (lat):",
                value=float(st.session_state["picked_lat"]),
                format="%.6f",
                key="manual_poly_lat",
            )
            manual_lon = mc2.number_input(
                "Долгота (lon):",
                value=float(st.session_state["picked_lon"]),
                format="%.6f",
                key="manual_poly_lon",
            )
            mc3.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if mc3.button("➕ Добавить узел", use_container_width=True):
                lat_val = round(float(manual_lat), 6)
                lon_val = round(float(manual_lon), 6)
                st.session_state["picked_lat"] = lat_val
                st.session_state["picked_lon"] = lon_val
                save_perimeter_polygon(polygon_points + [[lat_val, lon_val]])
                st.rerun()

            if len(polygon_points) < 3:
                st.warning(
                    f"Сейчас задано точек: **{len(polygon_points)}**. Для замыкания периметра добавьте ещё минимум **{3 - len(polygon_points)}** точк(и) кликом по карте или вручную."
                )
            else:
                st.success(f"Контур замкнут (вершин: **{len(polygon_points)}**).")

            # Редактируемый список всех вершин периметра с индивидуальным удалением
            if polygon_points:
                edited_vertices = []
                coords_changed = False
                for idx, pt_coords in enumerate(polygon_points):
                    v_c0, v_c1, v_c2, v_c3 = st.columns([0.45, 1.05, 1.05, 0.45])
                    v_c0.markdown(f"**#{idx + 1}**")
                    new_v_lat = v_c1.number_input(
                        f"lat_{idx}",
                        value=float(pt_coords[0]),
                        format="%.6f",
                        label_visibility="collapsed",
                        key=f"v_lat_{idx}_{len(polygon_points)}",
                    )
                    new_v_lon = v_c2.number_input(
                        f"lon_{idx}",
                        value=float(pt_coords[1]),
                        format="%.6f",
                        label_visibility="collapsed",
                        key=f"v_lon_{idx}_{len(polygon_points)}",
                    )
                    if (
                        round(new_v_lat, 6) != round(float(pt_coords[0]), 6)
                        or round(new_v_lon, 6) != round(float(pt_coords[1]), 6)
                    ):
                        coords_changed = True
                    edited_vertices.append([round(float(new_v_lat), 6), round(float(new_v_lon), 6)])

                    if v_c3.button("🗑️", key=f"del_poly_pt_{idx}", help=f"Удалить узел #{idx + 1}"):
                        new_pts = [p for i, p in enumerate(polygon_points) if i != idx]
                        save_perimeter_polygon(new_pts)
                        st.session_state["last_click_sig"] = None
                        st.rerun()

                if coords_changed:
                    if st.button("💾 Сохранить измененные координаты узлов", type="primary", use_container_width=True):
                        save_perimeter_polygon(edited_vertices)
                        st.rerun()

        # 2. Добавление новой камеры
        with st.expander("➕ Добавить новую камеру видеонаблюдения", expanded=True):
            with st.form("new_camera_form"):
                nc_name = st.text_input("Название рубежа:", placeholder="Например: Рубеж №4 — Западный пост")
                nc_desc = st.text_input("Описание сектора:", placeholder="Например: Сектор хранения оборудования")
                c_lat, c_lon = st.columns(2)
                nc_lat = c_lat.number_input("Широта (Latitude):", value=float(st.session_state["picked_lat"]), format="%.6f")
                nc_lon = c_lon.number_input("Долгота (Longitude):", value=float(st.session_state["picked_lon"]), format="%.6f")
                nc_proto = st.selectbox("Тип подключения:", ["RTSP", "HTTP/MJPEG", "USB/Local", "Video File"])
                nc_src = st.text_input(
                    "Источник видеопотока:",
                    value="rtsp://admin:12345@192.168.1.100:554/Streaming/Channels/101",
                )
                if st.form_submit_button("Разместить рубеж на карте", type="primary", use_container_width=True):
                    if nc_name.strip():
                        st.session_state["picked_lat"] = round(float(nc_lat), 6)
                        st.session_state["picked_lon"] = round(float(nc_lon), 6)
                        add_camera(nc_name.strip(), nc_desc.strip(), nc_lat, nc_lon, nc_proto, nc_src.strip())
                        st.success(f"Рубеж «{nc_name}» добавлен!")
                        st.rerun()
                    else:
                        st.error("Укажите название рубежа.")

    st.markdown("#### Подключенные рубежи видеонаблюдения")
    for cam in cameras_list:
        with st.container(border=True):
            cc1, cc2, cc3, cc4 = st.columns([1.5, 1.5, 1.2, 0.6])
            cc1.markdown(f"**{cam.name}**\n\n<small>{cam.sector_description or ''}</small>", unsafe_allow_html=True)
            cc2.code(f"[{cam.protocol_type}] {cam.video_source}", language="text")
            cc3.write(f"📍 `{cam.latitude:.5f}, {cam.longitude:.5f}`")
            if len(cameras_list) > 1:
                if cc4.button("Удалить", key=f"del_cam_{cam.id}"):
                    delete_camera(cam.id)
                    st.rerun()

with tab_log:
    st.subheader("Архив и ревью всех зафиксированных инцидентов")

    f_col1, f_col2 = st.columns([1.2, 1.0])
    with f_col1:
        filter_prio = st.multiselect(
            "Фильтр по уровню критичности:",
            options=["CRITICAL", "HIGH", "LOW"],
            default=["CRITICAL", "HIGH", "LOW"],
        )
    with f_col2:
        review_filter = st.radio(
            "Статус обработки (ревью):",
            options=["all", "unack", "ack"],
            format_func=lambda x: {
                "all": "Все записи",
                "unack": "⏳ Требуют ревью",
                "ack": "✅ Отработанные",
            }[x],
            horizontal=True,
        )

    filtered_incs = [i for i in incidents_list if i.alert_priority in filter_prio]
    if review_filter == "unack":
        filtered_incs = [i for i in filtered_incs if not i.is_acknowledged]
    elif review_filter == "ack":
        filtered_incs = [i for i in filtered_incs if i.is_acknowledged]

    st.caption(f"Отображено инцидентов: **{len(filtered_incs)}** из **{len(incidents_list)}**")

    if not filtered_incs:
        st.info("Записей по выбранному фильтру не найдено.")
    else:
        for inc in filtered_incs:
            with st.container(border=True):
                c1, c2, c3 = st.columns([1.2, 0.7, 2.0])
                fp = BASE_DIR / inc.full_frame_path
                if fp.exists():
                    c1.image(str(fp), caption=f"Снимок #{inc.id}", use_container_width=True)
                if inc.face_crop_path and (BASE_DIR / inc.face_crop_path).exists():
                    c2.image(str(BASE_DIR / inc.face_crop_path), caption="Фрагмент лица", use_container_width=True)
                with c3:
                    badge = (
                        "🔴 **CRITICAL (Сокрытие лица / маска)**"
                        if inc.alert_priority == "CRITICAL"
                        else ("🟠 **HIGH (Посторонний субъект)**" if inc.alert_priority == "HIGH" else "🟢 **LOW (Сотрудник)**")
                    )
                    ack_status = "✅ Отработано оператором" if inc.is_acknowledged else "⏳ Требует ревью"
                    st.markdown(f"### Инцидент #{inc.id} — {badge}")
                    st.write(f"🕒 **Время:** {inc.timestamp.strftime('%d.%m.%Y %H:%M:%S')} | **Ревью:** {ack_status}")
                    st.write(
                        f"📍 **Рубеж:** {inc.camera.name} "
                        f"(`{inc.camera.latitude:.5f}, {inc.camera.longitude:.5f}`)"
                    )
                    emp_str = inc.employee.full_name if inc.employee else "Посторонний (отсутствует в БД)"
                    st.write(
                        f"👤 **Идентификация:** `{inc.person_status}` ({emp_str}) | "
                        f"**Косинусное сходство:** `{inc.similarity_score:.3f}` | "
                        f"**Маска:** `{'Да' if inc.has_mask else 'Нет'}`"
                    )

                    act1, act2, _ = st.columns([1.2, 1.0, 1.5])
                    if not inc.is_acknowledged:
                        if act1.button("✅ Подтвердить (Отработано)", key=f"ack_log_{inc.id}", use_container_width=True):
                            set_incident_acknowledged(inc.id, True)
                            st.rerun()
                    else:
                        if act1.button("↩️ Вернуть в тревоги", key=f"unack_log_{inc.id}", use_container_width=True):
                            set_incident_acknowledged(inc.id, False)
                            st.rerun()

                    if act2.button("🗑️ Удалить запись", key=f"del_log_{inc.id}", use_container_width=True):
                        delete_incident(inc.id)
                        st.rerun()

with tab_emp:
    col_add, col_list = st.columns([1, 1.5])
    with col_add:
        st.subheader("➕ Регистрация нового пропуска")
        with st.form("add_emp_form"):
            f_name = st.text_input("ФИО сотрудника", placeholder="Иванов Иван Иванович")
            f_pos = st.text_input("Должность / подразделение", placeholder="Старший смены охраны")
            uploaded_file = st.file_uploader("Эталонная фотография (анфас, JPG/PNG)", type=["jpg", "jpeg", "png"])
            submitted = st.form_submit_button("Построить биометрический вектор и сохранить")

            if submitted and uploaded_file is not None and f_name.strip():
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                emb, _ = detector.biometrics.extract_embedding(img_bgr)
                if emb is not None:
                    fname = f"emp_{int(time.time())}.jpg"
                    fpath = EMPLOYEES_DIR / fname
                    cv2.imwrite(str(fpath), img_bgr)
                    add_employee(f_name.strip(), f_pos.strip(), f"data/employees/{fname}", emb, True)
                    detector.biometrics.reload_employees()
                    st.success(f"Пропуск для {f_name} активирован!")
                    st.rerun()
                else:
                    st.error("На фотографии не удалось выделить лицо.")

    with col_list:
        st.subheader("📋 Реестр допущенного персонала")
        if not employees_list:
            st.info("Реестр пуст. Добавьте сотрудника слева или снимите эталон с камеры в боковой панели.")
        for emp in employees_list:
            with st.container(border=True):
                ec1, ec2, ec3 = st.columns([0.7, 2.0, 0.6])
                epath = BASE_DIR / emp["photo_path"]
                if epath.exists():
                    ec1.image(str(epath), use_container_width=True)
                with ec2:
                    st.markdown(f"**{emp['full_name']}**")
                    st.write(f"Должность: {emp['position']}")
                    st.caption(f"Биометрический шаблон: ArcFace 512-D Float32 | ID: {emp['id']}")
                with ec3:
                    if st.button("Удалить", key=f"del_emp_{emp['id']}"):
                        delete_employee(emp["id"])
                        detector.biometrics.reload_employees()
                        st.rerun()

with tab_analytics:
    st.subheader("📊 Распределение инцидентов по рубежам охраны")
    cam_counts = {c.name: 0 for c in cameras_list}
    for inc in incidents_list:
        if inc.camera:
            cam_counts[inc.camera.name] = cam_counts.get(inc.camera.name, 0) + 1
    st.bar_chart(cam_counts)

    st.subheader("⚙️ Телеметрия вычислительного ядра и нейросетевых модулей")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        st.info(f"**Вычислительный узел:** {gpu_name} | **Занято VRAM (PyTorch):** {vram_alloc:.1f} МБ из {vram_total:.1f} ГБ")

    st.table([
        {
            "Подсистема": "Детекция объектов и трекинг траекторий",
            "Архитектура": "YOLO11s + ByteTrack",
            "Точность": "47.0% mAP (COCO)",
            "Задержка инференса": "~3.8 мс / кадр",
        },
        {
            "Подсистема": "Биометрическая верификация («Свой — Чужой»)",
            "Архитектура": "InsightFace (RetinaFace + ArcFace R50)",
            "Точность": "99.83% (LFW)",
            "Задержка инференса": "~6.5 мс (по триггеру пересечения)",
        },
        {
            "Подсистема": "Детекция средств сокрытия лица (маски / балаклавы)",
            "Архитектура": "Анализ ключевых точек и хроматических зон YCrCb",
            "Точность": "96.4%",
            "Задержка инференса": "< 0.4 мс",
        },
    ])