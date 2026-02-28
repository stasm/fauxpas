/*
    Matrix utility functions

    Note that OpenGL expects column-major arrays, but JavaScript, is row-major.
    So each matrix in code is written as the transpose of its mathematical form.
*/
export function mtx_zero() {
    return [
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    ];
}

export function mtx_identity() {
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ];
}

export function mtx_mul(a, b) {
    var c = mtx_zero();

    for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 4; j++) {
            for (var k = 0; k < 4; k++) {
                c[i + j * 4] += a[i + k * 4] * b[k + j * 4];
            }
        }
    }

    return c;
}

export function mtx_transpose(a) {
    var b = mtx_zero();

    for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 4; j++) {
            b[i + j * 4] = a[j + i * 4];
        }
    }

    return b;
}

export function mtx_inverse(m) {
    var inv = mtx_zero();
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    var det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0) {
        console.log("Error: Non-invertible matrix");
        return mtx_zero();
    }

    det = 1.0 / det;
    for (var i = 0; i < 16; i++) {
        inv[i] *= det;
    }
    return inv;
}

export function mtx_translation(x, y, z) {
    return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        x, y, z, 1
    ];
}

export function mtx_rotation_x(r) {
    return [
        1, 0, 0, 0,
        0, Math.cos(r), Math.sin(r), 0,
        0, -Math.sin(r), Math.cos(r), 0,
        0, 0, 0, 1
    ];
}

export function mtx_rotation_y(r) {
    return [
        Math.cos(r), 0, -Math.sin(r), 0,
        0, 1, 0, 0,
        Math.sin(r), 0, Math.cos(r), 0,
        0, 0, 0, 1
    ];
}

export function mtx_perspective(fov_y, aspect, z_near, z_far) {
    var top = z_near * Math.tan(fov_y * Math.PI / 360.0);
    var bot = -top;
    var left = bot * aspect;
    var right = top * aspect;

    var X = 2 * z_near / (right - left);
    var Y = 2 * z_near / (top - bot);
    var A = (right + left) / (right - left);
    var B = (top + bot) / (top - bot);
    var C = -(z_far + z_near) / (z_far - z_near);
    var D = -2 * z_far * z_near / (z_far - z_near);

    return [
        X, 0.0, 0.0, 0.0,
        0.0, Y, 0.0, 0.0,
        A, B, C, -1.0,
        0.0, 0.0, D, 0.0
    ];
}
