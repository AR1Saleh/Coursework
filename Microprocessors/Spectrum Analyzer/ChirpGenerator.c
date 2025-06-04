#include "TM4C123GH6PM.h"
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

// Constants
#define SYSCLK         16000000UL  // 16 MHz
#define TIMER_FREQ     1000UL      // 1 ms steps
#define CHIRP_DURATION 60.0f       // 60 seconds
#define CHIRP_F0       1.0f        // start freq (1Hz)
#define CHIRP_F1       10.0f       // end freq (10Hz)
#define CHIRP_AMPL     1.65f       // amplitude for 3.3Vpp
#define PI             3.141592653589793f
#define SAMPLE_COUNT   60000       // 60s * 1000 samples/s
#define FILTER_SAMPLES 3           // Minor software averaging

// Port masks
#define START_BTN_MASK  (1U << 0)   // PF0 (SW2)
#define ABORT_BTN_MASK  (1U << 4)   // PF4 (SW1)
#define BUSY_LED_MASK   (1U << 2)   // PF2 (Blue LED)

// System states
typedef enum {
    STATE_IDLE,
    STATE_ACQUIRING,
    STATE_ABORTED,
    STATE_COMPLETE,
    STATE_FAULT
} system_state_t;

// Function prototypes
void sendString(const char *str);
void sendFloat(float value);
void initUART(void);
void initGPIO(void);
void initADC(void);
void initPWM(void);
void initTimer0(void);
void initSysTick(void);
void sendDataPacket(float u, float y);

// Global variables
volatile uint32_t g_ms_count = 0;
volatile system_state_t g_state = STATE_IDLE;
volatile uint32_t g_sample_count = 0;
volatile float g_current_u = 0.0f;

// SysTick Interrupt Handler
void SysTick_Handler(void) {
    g_ms_count++;
}

// Timer0A Interrupt Handler
void TIMER0A_Handler(void) {
    float t, phase;
    uint32_t duty, i, adc_value;
    uint32_t adc_sum = 0;
    float y;
    
    TIMER0->ICR |= (1 << 0);  // Clear timeout interrupt
    
    if(g_state != STATE_ACQUIRING) return;
    
    // Calculate chirp signal
    t = g_sample_count * 0.001f;
    phase = 2 * PI * (CHIRP_F0 * t + 0.5f * (CHIRP_F1 - CHIRP_F0) * t * t / CHIRP_DURATION);
    g_current_u = 1.65f + CHIRP_AMPL * sinf(phase);
    
    // Update PWM output
    duty = (uint32_t)((g_current_u / 3.3f) * 1600.0f);
    if(duty > 1599) duty = 1599;
    PWM0->_0_CMPA = duty;
    
    // Take multiple ADC samples for stability
    for(i = 0; i < FILTER_SAMPLES; i++) {
        ADC0->PSSI |= (1 << 0);
        while(!(ADC0->RIS & (1 << 0))) {}  // Wait for conversion
        adc_value = ADC0->SSFIFO0;
        ADC0->ISC = (1 << 0);
        adc_sum += adc_value;
    }
    
    // Calculate average voltage
    y = (3.3f * (adc_sum / FILTER_SAMPLES)) / 4095.0f;
    
    // Send data to PC
    sendDataPacket(g_current_u, y);
    
    // Update sample count
    g_sample_count++;
    
    // Check completion
    if(g_sample_count >= SAMPLE_COUNT) {
        g_state = STATE_COMPLETE;
    }
}

// ----------------------------------------------------------------------------
int main(void) {
    uint32_t last_start_time = 0, last_abort_time = 0, now;
    bool start_pressed = false, abort_pressed = false;
    
    // Initialize hardware
    SystemInit();
    initUART();
    initGPIO();
    initADC();
    initPWM();
    initTimer0();
    initSysTick();
    
    sendString("System Ready. Press START.\n");

    while(1) {
        now = g_ms_count;
        start_pressed = false;
        abort_pressed = false;
        
        // START button (PF0)
        if(!(GPIOF->DATA & START_BTN_MASK) && (now - last_start_time > 20)) {
            start_pressed = true;
            last_start_time = now;
        }
        
        // ABORT button (PF4)
        if(!(GPIOF->DATA & ABORT_BTN_MASK) && (now - last_abort_time > 20)) {
            abort_pressed = true;
            last_abort_time = now;
        }
        
        // State machine
        switch(g_state) {
            case STATE_IDLE:
                if(start_pressed) {
                    g_state = STATE_ACQUIRING;
                    g_sample_count = 0;
                    GPIOF->DATA |= BUSY_LED_MASK;
                    sendString("ACQUIRING\n");
                    PWM0->ENABLE |= (1 << 0);
                    TIMER0->CTL |= (1 << 0);
                }
                break;
                
            case STATE_ACQUIRING:
                if(abort_pressed) g_state = STATE_ABORTED;
                break;
                
            case STATE_ABORTED:
                TIMER0->CTL &= ~(1 << 0);
                PWM0->ENABLE &= ~(1 << 0);
                GPIOF->DATA &= ~BUSY_LED_MASK;
                sendString("ABORTED\n");
                g_state = STATE_IDLE;
                break;
                
            case STATE_COMPLETE:
                TIMER0->CTL &= ~(1 << 0);
                PWM0->ENABLE &= ~(1 << 0);
                GPIOF->DATA &= ~BUSY_LED_MASK;
                sendString("COMPLETE\n");
                g_state = STATE_IDLE;
                break;
                                                        
            case STATE_FAULT:
                TIMER0->CTL &= ~(1 << 0);
                PWM0->ENABLE &= ~(1 << 0);
                GPIOF->DATA &= ~BUSY_LED_MASK;
                sendString("FAULT\n");
                while(1); // Halt
        }
    }
}

// ----------------------------------------------------------------------------
// Hardware Initialization Functions
void initUART(void) {
    // Enable clocks
    SYSCTL->RCGCGPIO |= (1 << 0);  // GPIOA
    SYSCTL->RCGCUART |= (1 << 0);  // UART0
    
    // Configure PA0 (RX), PA1 (TX)
    GPIOA->AFSEL |= 0x03;
    GPIOA->PCTL = (GPIOA->PCTL & ~0xFF) | 0x11;
    GPIOA->DEN |= 0x03;
    
    // UART configuration: 115200 baud, 8N1
    UART0->CTL = 0;         // Disable UART
    UART0->IBRD = 1;        // 16MHz/(16*1) = 1,000,000 baud
    UART0->FBRD = 0;       // Fractional
    UART0->LCRH = (3 << 5); // 8-bit, no parity
    UART0->CC = 0;          // System clock
    UART0->CTL = (1 << 0) | (1 << 8) | (1 << 9); // Enable UART, TX, RX
}

void initGPIO(void) {
    // Enable clock
    SYSCTL->RCGCGPIO |= (1 << 5); // GPIOF

    // Unlock and configure PF0, PF4 (buttons), PF2 (LED), PF3 (Debug)
    GPIOF->LOCK = 0x4C4F434B; // Unlock
    GPIOF->CR = 0x1F;         // Enable commit
    GPIOF->LOCK = 0;          // Relock
    
    GPIOF->DIR &= ~(START_BTN_MASK | ABORT_BTN_MASK); // Inputs
    GPIOF->DIR |= BUSY_LED_MASK ;          // Outputs
    GPIOF->DEN |= START_BTN_MASK | ABORT_BTN_MASK | BUSY_LED_MASK;
    GPIOF->PUR |= START_BTN_MASK | ABORT_BTN_MASK;    // Pull-ups
    GPIOF->DATA &= ~(BUSY_LED_MASK);      // LEDs off
}

void initADC(void) {
    // Enable clocks
    SYSCTL->RCGCGPIO |= (1 << 4); // GPIOE
    SYSCTL->RCGCADC |= (1 << 0);  // ADC0
    
    // Configure PE3 (AIN0) as analog input
    GPIOE->AFSEL &= ~(1 << 3);
    GPIOE->AMSEL |= (1 << 3);
    GPIOE->DEN &= ~(1 << 3);
    
    // ADC configuration
    ADC0->ACTSS &= ~(1 << 0);  // Disable sample sequencer 0
    ADC0->EMUX &= ~0xF;        // Software trigger
    ADC0->SSMUX0 = 0;          // Channel AIN0 (PE3)
    ADC0->SSCTL0 = (1 << 1) | (1 << 2); // IE0, END0
    ADC0->ACTSS |= (1 << 0);   // Enable sequencer 0
}

void initPWM(void) {
    // Enable clocks
    SYSCTL->RCGCGPIO |= (1 << 1); // GPIOB
    SYSCTL->RCGCPWM |= (1 << 0);  // PWM0
    
    // Configure PB6 (M0PWM0)
    GPIOB->AFSEL |= (1 << 6);
    GPIOB->PCTL = (GPIOB->PCTL & ~0x0F000000) | (0x04 << 24);
    GPIOB->DEN |= (1 << 6);
    
    // PWM configuration (10kHz frequency)
    PWM0->_0_CTL = 0;        // Disable generator
    PWM0->_0_GENA = 0x0000008C; // Drive high on load, low on compare
    PWM0->_0_LOAD = 1600 - 1;  // 16MHz / 1600 = 10kHz
    PWM0->_0_CMPA = 800 - 1;   // 50% initial duty
    PWM0->_0_CTL = (1 << 0); // Enable generator
    PWM0->ENABLE = 0;        // Disable output initially
}

void initTimer0(void) {
    // Enable clock
    SYSCTL->RCGCTIMER |= (1 << 0);
    
    // Configure Timer0A for 1ms interrupts
    TIMER0->CTL = 0;         // Disable timer
    TIMER0->CFG = 0x00;      // 32-bit mode
    TIMER0->TAMR = 0x02;     // Periodic mode
    TIMER0->TAILR = 16000 - 1; // 1ms at 16MHz
    TIMER0->IMR |= (1 << 0); // Enable timeout interrupt
    
    // Configure timer interrupt
    NVIC->ISER[0] |= (1 << 19); // Enable Timer0A interrupt
    NVIC->IP[19] = 0x00;        // Highest priority
}

void initSysTick(void) {
    SysTick->LOAD = 16000 - 1; // 1ms at 16MHz
    SysTick->VAL = 0;
    SysTick->CTRL = (1 << 2) | (1 << 1) | (1 << 0); // System clock, interrupt, enable
}

// Data Transmission Functions
void sendString(const char *str) {
    while(*str) {
        while(UART0->FR & (1 << 5));
        UART0->DR = *str++;
    }
}

void sendFloat(float value) {
    uint8_t *bytes = (uint8_t *)&value;
    int i;
    for(i = 0; i < 4; i++) {
        while(UART0->FR & (1 << 5));
        UART0->DR = bytes[i];
    }
}

void sendDataPacket(float u, float y) {
    sendFloat(u);
    sendFloat(y);
}

// System Initialization
void SystemInit(void) {
    __disable_irq();
    SCB->CPACR |= 0x00F00000; // Enable FPU
    __enable_irq();
}
