# Algorithms

The project applies:

- Sobel operator for gradient magnitude
- Laplacian operator for second-order derivatives
- Binary thresholding (`auto` via Otsu or fixed integer threshold)
- Contour extraction with `cv2.findContours`

The contour stage is applied to the binary edge map.

## Sobel

The Sobel operator estimates first-order derivatives in X and Y directions.
The final edge strength map is calculated as gradient magnitude:

```text
sqrt(gx^2 + gy^2)
```

## Laplacian

The Laplacian operator estimates second-order derivatives and highlights
rapid intensity changes. In the implementation, absolute response values are
normalized to `uint8`.

## Thresholding

Two modes are supported:

- Fixed integer threshold (`0..255`)
- Otsu automatic threshold (`auto`)

## Contours

Contours are extracted from the binary edge map, then optional filtering by
minimum area is applied (`--min-contour-area`).
