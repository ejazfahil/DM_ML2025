"""Thermal preprocessing. 2025-05-02"""
import numpy as np

def normalize(img, vmin=None, vmax=None):
    vmin = vmin or img.min(); vmax = vmax or img.max()
    return (img-vmin)/(vmax-vmin+1e-8)

def threshold_hotspots(img, pct=95):
    return (img > np.percentile(img, pct)).astype(np.uint8)
