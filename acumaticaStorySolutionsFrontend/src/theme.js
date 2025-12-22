import { createTheme } from '@mui/material';

const getTheme = (mode) => ({
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 960,
      lg: 1280,
      xl: 1920,
    },
  },
  palette: {
    mode,
    primary: {
      main: '#0075C9', // Acumatica Blue
      light: '#49A3DC', // Soft Sky Blue
      dark: '#005B9F', // Darker shade for hover
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#49A3DC', // Soft Sky Blue
      light: '#E6F3FB', // Light Azure
      dark: '#3B82F6', // Darker blue for emphasis
      contrastText: '#FFFFFF',
    },
    background: {
      default: '#E6F3FB', // Light Azure - matching the global background
      paper: '#FFFFFF', // Pure White
    },
    text: {
      primary: '#333333', // Dark Slate Gray
      secondary: '#6B7280', // Medium Gray
    },
    success: {
      main: '#28C76F', // Soft Green
      light: '#E6F9F0',
      dark: '#1F9D57',
    },
    error: {
      main: '#EA5455', // Soft Red
      light: '#FEECEC',
      dark: '#BB4344',
    },
    warning: {
      main: '#FF9F43',
      light: '#FFF4EB',
      dark: '#CC7F36',
    },
    info: {
      main: '#49A3DC',
      light: '#E6F3FB',
      dark: '#3B82F6',
    },
    divider: mode === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.08)',
    custom: {
      botHighlight: '#E6F3FB', // Light Azure for bot bubbles
      darkNavy: '#1E293B', // Dark mode background
    },
  },
  shape: {
    borderRadius: 12,
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", "Helvetica Neue", -apple-system, sans-serif',
    htmlFontSize: 16,
    fontSize: 14,
    h1: {
      fontWeight: 600,
      fontSize: 'clamp(2rem, 5vw, 3.5rem)',
      lineHeight: 1.2,
    },
    h2: {
      fontWeight: 600,
      fontSize: 'clamp(1.75rem, 4vw, 3rem)',
      lineHeight: 1.3,
    },
    h3: {
      fontWeight: 600,
      fontSize: 'clamp(1.5rem, 3.5vw, 2.5rem)',
      lineHeight: 1.3,
    },
    h4: {
      fontWeight: 600,
      fontSize: 'clamp(1.25rem, 3vw, 2rem)',
      lineHeight: 1.4,
    },
    h5: {
      fontWeight: 600,
      fontSize: 'clamp(1.1rem, 2.5vw, 1.5rem)',
      lineHeight: 1.4,
    },
    h6: {
      fontWeight: 600,
      fontSize: 'clamp(1rem, 2vw, 1.25rem)',
      lineHeight: 1.5,
    },
    subtitle1: {
      fontSize: 'clamp(0.9rem, 1.5vw, 1.1rem)',
      fontWeight: 500,
      lineHeight: 1.5,
    },
    subtitle2: {
      fontSize: 'clamp(0.85rem, 1.25vw, 1rem)',
      fontWeight: 500,
      lineHeight: 1.5,
    },
    body1: {
      fontSize: 'clamp(0.875rem, 1.25vw, 1rem)',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: 'clamp(0.8rem, 1.1vw, 0.9rem)',
      lineHeight: 1.6,
    },
    caption: {
      fontSize: 'clamp(0.7rem, 1vw, 0.8rem)',
      lineHeight: 1.5,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
      fontSize: 'clamp(0.85rem, 1.25vw, 1rem)',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '8px 16px',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-1px)',
          },
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 8px rgba(0, 117, 201, 0.15)',
          },
        },
        outlined: {
          borderWidth: '1.5px',
          '&:hover': {
            borderWidth: '1.5px',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
        rounded: {
          borderRadius: 12,
        },
        elevation1: {
          boxShadow: mode === 'light' 
            ? '0px 2px 4px rgba(0, 0, 0, 0.05), 0px 1px 2px rgba(0, 0, 0, 0.08)'
            : '0px 2px 4px rgba(0, 0, 0, 0.2), 0px 1px 2px rgba(0, 0, 0, 0.15)',
        },
        elevation2: {
          boxShadow: mode === 'light'
            ? '0px 4px 8px rgba(0, 0, 0, 0.08), 0px 2px 4px rgba(0, 0, 0, 0.1)'
            : '0px 4px 8px rgba(0, 0, 0, 0.3), 0px 2px 4px rgba(0, 0, 0, 0.2)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              borderColor: '#49A3DC',
            },
            '&.Mui-focused': {
              boxShadow: '0 0 0 3px rgba(0, 117, 201, 0.15)',
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          height: 28,
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
        standardSuccess: {
          backgroundColor: '#E6F9F0',
          color: '#1F9D57',
        },
        standardError: {
          backgroundColor: '#FEECEC',
          color: '#BB4344',
        },
        standardWarning: {
          backgroundColor: '#FFF4EB',
          color: '#CC7F36',
        },
        standardInfo: {
          backgroundColor: '#E6F3FB',
          color: '#0075C9',
        },
      },
    },
  },
});

const theme = createTheme(getTheme('light'));

export default theme;

