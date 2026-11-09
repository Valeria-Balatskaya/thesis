# src/attack_suite.py
# The 23-attack suite used for the thesis' main robustness numbers
# (12 classic + 11 modern; identical to the list in tests/test_v3_vs_v2.py, which
# produced the v3 results). Each entry: (name, function(src, dst, **kw) or None, kwargs).
# GEOMETRIC = attacks that a position-dependent blind decoder cannot handle by design
# (they need Retrieve-Align-Verify).

from src import attacks, ecommerce_attacks as ea, modern_attacks as ma

ATTACKS = [
    ("clean",              None,                              {}),
    ("jpeg_q70",           attacks.jpeg_compress,             {"quality": 70}),
    ("jpeg_q30",           attacks.jpeg_compress,             {"quality": 30}),
    ("resize_0.5",         attacks.resize_attack,             {"scale": 0.5}),
    ("noise_s15",          attacks.gaussian_noise,            {"sigma": 15}),
    ("crop_0.1",           attacks.crop_attack,               {"crop_fraction": 0.1}),
    ("brightness_1.2",     attacks.brightness_attack,         {"factor": 1.2}),
    ("marketplace_square", ea.marketplace_square,             {"target_size": 512}),
    ("instagram_filter",   ea.instagram_filter,               {}),
    ("screenshot",         ea.screenshot_simulation,          {}),
    ("print_photograph",   ea.print_photograph_simulation,    {}),
    ("native_screenshot",  ea.native_screenshot_simulation,   {}),
    ("social_recompress",  ma.social_recompress,              {}),
    ("cdn_webp_360",       ma.marketplace_variant,            {"size": 360}),
    ("offcentre_crop_70",  ma.offcentre_crop,                 {"keep": 0.7, "dx": 1.0, "dy": 0.0}),
    ("offcentre_crop_50",  ma.offcentre_crop,                 {"keep": 0.5, "dx": 0.2, "dy": 0.9}),
    ("rotate_8",           ma.rotate,                         {"degrees": 8}),
    ("mirror",             ma.mirror,                         {}),
    ("perspective",        ma.perspective,                    {"strength": 0.12}),
    ("page_screenshot",    ma.page_screenshot,                {}),
    ("screen_photo",       ma.screen_photo,                   {}),
    ("promo_overlay",      ma.promo_overlay,                  {}),
    ("inpaint_region",     ma.inpaint_region,                 {}),
]
GEOMETRIC = {"offcentre_crop_70", "offcentre_crop_50", "rotate_8", "mirror", "perspective",
             "page_screenshot"}
