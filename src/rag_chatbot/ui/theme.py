from gradio.themes import Base, GoogleFont

primary = "#AAB9DA"
secondary = "#7BE0C3"
tertiary = "#F29BC4"
primary_background = "#B3E7F2"
secondary_background = "#F1FAFE"
primary_text = "#292f46"

def get_css():
    css="""
        footer {display: none !important;}
        #company-logo { 
            display:flex; 
            justify-content:flex-end; 
            align-items:center; 
            background: var(--background-fill-primary) !important; 
            padding: 0; 
            margin: 0; 
            border: none !important; 
            box-shadow: none !important; 
            outline: none !important; 
            border-color: var(--background-fill-primary) !important;
        }
        #company-logo img { display:block; }
    """
    return css

def get_theme():
    return Base(
        text_size="md",
        spacing_size="md",
        radius_size="md",
        font=[GoogleFont("Inter"), "Inter", "system-ui", "Segoe UI", "Roboto", "sans-serif"]
        ).set(
                    # Text
        body_text_color=primary_text,
        block_title_text_color=primary_text,
        block_label_text_color=primary_text,

        # Backgrounds (single background applied broadly)
        background_fill_primary=primary_background,
        background_fill_secondary=secondary_background,
        block_background_fill=secondary_background,
        code_background_fill=secondary_background,
        stat_background_fill=secondary_background,

        # Borders (reuse TEXT to stay within 3-color rule)
        border_color_primary=primary_text,
        border_color_accent=primary_text,
        block_border_color=primary_text,
        table_border_color=primary_text,

        # Links use TEXT color
        link_text_color=primary_text,
        link_text_color_hover=primary_text,
        link_text_color_active=primary_text,
        link_text_color_visited=primary_text,

        # Flat look
        block_shadow="none",
        color_accent_soft=primary_background,

        # Inputs
        input_background_fill=secondary_background,
        input_background_fill_focus="none",
        input_border_color_focus=primary_text,
        input_placeholder_color=primary_text,
        input_shadow="none",
        input_shadow_focus="none",

        # Sliders and selection accents use BUTTON
        slider_color=primary,

        # Checkboxes / toggles
        checkbox_background_color=primary,
        checkbox_background_color_selected=primary,
        checkbox_border_color=primary_text,
        checkbox_border_color_hover=primary_text,
        checkbox_label_background_fill=primary_background,
        checkbox_label_background_fill_selected=primary_background,
        checkbox_label_border_color_selected=primary_text,
        checkbox_label_text_color_selected_dark=primary_text,

        # Buttons
        button_primary_background_fill=primary,
        button_primary_background_fill_hover=secondary,
        button_primary_border_color=primary,
        button_primary_text_color=primary_text,
        button_primary_shadow="none",
        button_primary_shadow_hover="none",
        button_primary_shadow_active="none",

        button_secondary_background_fill=primary,
        button_secondary_background_fill_hover=secondary,
        button_secondary_border_color=primary,
        button_secondary_border_color_hover=primary,
        button_secondary_text_color=primary_text,
        button_secondary_shadow="none",
        button_secondary_shadow_hover="none",
        button_secondary_shadow_active="none",

        # Labels / chips align with backgrounds
        block_label_background_fill=primary_background,

        # Errors reuse our 3 colors
        error_background_fill=primary_background,
        error_border_color=primary,
        error_text_color=primary_text,
        error_icon_color=primary,
        )