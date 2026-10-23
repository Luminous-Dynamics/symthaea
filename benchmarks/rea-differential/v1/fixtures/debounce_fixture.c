/*
 * Synthetic firmware-like debounce fixture for ENG-REA-002.
 * This is a host-built test fixture, not production firmware and not for a device.
 * Build-time threshold and timer configuration are controlled by the corpus manifest.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DEBOUNCE_THRESHOLD_TICKS
#define DEBOUNCE_THRESHOLD_TICKS 12
#endif

static int accept_pulse(unsigned long pulse_ticks) {
    return pulse_ticks >= (unsigned long)DEBOUNCE_THRESHOLD_TICKS;
}

int main(int argc, char **argv) {
    char *end = NULL;
    unsigned long ticks;

    if (argc != 2) {
        fputs("usage: debounce_fixture PULSE_TICKS\n", stderr);
        return 64;
    }

    if (argv[1][0] == '\0') {
        fputs("PULSE_TICKS must be a non-negative decimal integer\n", stderr);
        return 65;
    }
    for (const char *p = argv[1]; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            fputs("PULSE_TICKS must be a non-negative decimal integer\n", stderr);
            return 65;
        }
    }

    errno = 0;
    ticks = strtoul(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || *end != '\0') {
        fputs("PULSE_TICKS must be a non-negative decimal integer\n", stderr);
        return 65;
    }

    puts(accept_pulse(ticks) ? "ACCEPT" : "REJECT");
    return 0;
}
