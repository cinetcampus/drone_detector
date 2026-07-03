"""Measurement packet builder compatible with measurement.h/cpp layout.

Packet layout:
- byte 0: header (version + packet type)
- byte 1: payload length (bytes after length field)
- bytes 2..N: measurement entries
"""

from enum import IntEnum


MEASUREMENT_HEADER_LENGTH = 2
MAX_PAYLOAD_LENGTH = 0xFF


class PacketVersion(IntEnum):
    VERSION_1_0 = 0x00


class PacketType(IntEnum):
    MEASUREMENT_DATA = 0x01


class MeasurementType(IntEnum):
    TEMPERATURE = 0x00
    HUMIDITY = 0x01
    LUMINOSITY = 0x02
    NOISE_LEVEL = 0x03
    AMMONIAC = 0x04
    ACTIVITY = 0x05
    OBSERVATIONS = 0x06
    RENKEER_WEATHER_STATION = 0x07
    LUFFT_WS100 = 0x08
    SEN54_DUST = 0x09
    DUST_COMPARISON = 0x0A
    BATTERY_VOLTAGE = 0x0B
    DUST_CALYPSO = 0x0C
    CALYPSO_DATA = 0x0D
    DUST_FULL = 0x0E
    AQT560 = 0x0F
    LCJ_CAPTEURS_CV7 = 0x10
    AIRMAR_WX150 = 0x11
    NEIGHBORTABLE = 0x12
    SEN54_DUST_MUX = 0x13
    DEVICE_STATUS = 0x1F


class MeasurementFormat(IntEnum):
    SINGLE_VALUE = 0x00
    MIN_MAX_MEAN = 0x01
    TIME_SERIES = 0x02


def measurement_header(version: PacketVersion) -> int:
    return ((int(version) & 0x0F) << 4) | (int(PacketType.MEASUREMENT_DATA) & 0x0F)


def measurement_type(measurement_type_value: MeasurementType, fmt: MeasurementFormat) -> int:
    return ((int(measurement_type_value) & 0x1F) << 3) | (int(fmt) & 0x07)


class MeasurementPacket:
    def __init__(self, version: PacketVersion = PacketVersion.VERSION_1_0):
        self._buffer = bytearray([measurement_header(version), 0])

    def _append_type(self, measurement_type_value: MeasurementType, fmt: MeasurementFormat):
        self._buffer.append(measurement_type(measurement_type_value, fmt))

    def _set_length(self):
        # Payload length excludes first two header bytes.
        payload_len = len(self._buffer) - MEASUREMENT_HEADER_LENGTH
        if payload_len > MAX_PAYLOAD_LENGTH:
            raise ValueError(
                f"Measurement payload too long: {payload_len} bytes (max {MAX_PAYLOAD_LENGTH})"
            )
        self._buffer[1] = payload_len

    def add_value8(
        self,
        measurement_type_value: MeasurementType,
        fmt: MeasurementFormat,
        value: int,
    ):
        self._append_type(measurement_type_value, fmt)
        self._buffer.append(int(value) & 0xFF)
        self._set_length()

    def add_value16(
        self,
        measurement_type_value: MeasurementType,
        fmt: MeasurementFormat,
        value: int,
    ):
        self._append_type(measurement_type_value, fmt)
        clamped = int(value) & 0xFFFF
        self._buffer.extend(clamped.to_bytes(2, byteorder="big", signed=False))
        self._set_length()

    def add_array8(
        self,
        measurement_type_value: MeasurementType,
        fmt: MeasurementFormat,
        values: list[int],
    ):
        self._append_type(measurement_type_value, fmt)
        if int(fmt) & int(MeasurementFormat.TIME_SERIES):
            if len(values) > 255:
                raise ValueError("TimeSeries length must be <= 255")
            self._buffer.append(len(values) & 0xFF)
        for value in values:
            self._buffer.append(int(value) & 0xFF)
        self._set_length()

    def to_bytes(self) -> bytes:
        return bytes(self._buffer)
