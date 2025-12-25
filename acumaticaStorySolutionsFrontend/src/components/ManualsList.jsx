import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert,
  useTheme,
  alpha,
  Collapse,
  IconButton,
} from '@mui/material';
import {
  MenuBook as MenuBookIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';
import { getManualsList } from '../api/solutions';

const ManualsList = ({ isExpanded: controlledExpanded, onToggle, autoCollapse = false }) => {
  const theme = useTheme();
  const [manuals, setManuals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [internalExpanded, setInternalExpanded] = useState(controlledExpanded !== undefined ? controlledExpanded : true);
  
  // Use controlled or internal state
  const isExpanded = controlledExpanded !== undefined ? controlledExpanded : internalExpanded;
  
  // Auto-collapse when form is submitted
  useEffect(() => {
    if (autoCollapse) {
      const newState = false;
      setInternalExpanded(newState);
      onToggle?.(newState);
    }
  }, [autoCollapse, onToggle]);
  
  const handleToggle = () => {
    const newState = !isExpanded;
    if (controlledExpanded === undefined) {
      setInternalExpanded(newState);
    }
    onToggle?.(newState);
  };

  const fetchManuals = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getManualsList();
      if (response.success) {
        setManuals(response.manuals || []);
      } else {
        setError(response.error || 'Failed to load manuals');
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch manuals list');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchManuals();
  }, []);

  return (
    <Paper
      elevation={2}
      sx={{
        height: isExpanded ? '100%' : 'auto',
        maxHeight: isExpanded ? '100%' : 'none',
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 2,
        overflow: 'hidden',
        bgcolor: theme.palette.background.paper,
        minHeight: 0,
      }}
    >
      {/* Collapsible Header */}
      <Box
        onClick={handleToggle}
        sx={{
          p: 2,
          borderBottom: isExpanded ? `1px solid ${theme.palette.divider}` : 'none',
          bgcolor: alpha(theme.palette.primary.main, 0.05),
          display: 'flex',
          alignItems: 'center',
          gap: 1.5,
          cursor: 'pointer',
          userSelect: 'none',
          transition: 'all 0.2s ease',
          flexShrink: 0,
          '&:hover': {
            bgcolor: alpha(theme.palette.primary.main, 0.08),
          },
        }}
      >
        <Box
          sx={{
            width: 36,
            height: 36,
            borderRadius: 1.5,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
            color: 'white',
            flexShrink: 0,
          }}
        >
          <MenuBookIcon sx={{ fontSize: 20 }} />
        </Box>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1rem' }}>
            Knowledge Base
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {isExpanded ? `${manuals.length} manuals available` : 'Click to expand'}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          {!loading && (
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                fetchManuals();
              }}
              sx={{
                p: 0.5,
                color: 'text.secondary',
                '&:hover': {
                  bgcolor: 'action.hover',
                  color: 'primary.main',
                },
              }}
              title="Refresh list"
            >
              <RefreshIcon sx={{ fontSize: 18 }} />
            </IconButton>
          )}
          <IconButton
            size="small"
            sx={{
              p: 0.5,
              color: 'text.secondary',
              transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
            }}
          >
            <ExpandMoreIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Collapsible Content Wrapper */}
      {isExpanded && (
        <Box
          sx={{
            flex: 1,
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
        <Collapse 
          in={isExpanded} 
          timeout={600} 
          easing="cubic-bezier(0.4, 0, 0.2, 1)"
          sx={{
            flex: 1,
            minHeight: 0,
            display: isExpanded ? 'flex' : 'block',
            flexDirection: 'column',
            width: '100%',
            '& .MuiCollapse-wrapper': {
              display: 'flex',
              flexDirection: 'column',
              height: isExpanded ? '100%' : 'auto',
            },
            '& .MuiCollapse-wrapperInner': {
              display: 'flex',
              flexDirection: 'column',
              flex: 1,
              minHeight: 0,
              height: isExpanded ? '100%' : 'auto',
            },
          }}
        >
          <Box
            sx={{
              flex: 1,
              overflowY: 'auto',
              overflowX: 'hidden',
              p: 1,
              minHeight: 0,
              maxHeight: isExpanded ? '100%' : 'none',
              display: 'flex',
              flexDirection: 'column',
              '&::-webkit-scrollbar': {
                width: '8px',
              },
              '&::-webkit-scrollbar-track': {
                background: 'transparent',
              },
              '&::-webkit-scrollbar-thumb': {
                background: alpha(theme.palette.primary.main, 0.2),
                borderRadius: '4px',
                '&:hover': {
                  background: alpha(theme.palette.primary.main, 0.3),
                },
              },
            }}
          >
        {loading && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 4,
              gap: 2,
            }}
          >
            <CircularProgress size={32} />
            <Typography variant="body2" color="text.secondary">
              Loading manuals...
            </Typography>
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ m: 1, borderRadius: 1.5 }}>
            {error}
          </Alert>
        )}

        {!loading && !error && manuals.length === 0 && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 4,
              px: 2,
            }}
          >
            <MenuBookIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
            <Typography variant="body2" color="text.secondary" align="center">
              No manuals found in knowledge base
            </Typography>
          </Box>
        )}

        {!loading && !error && manuals.length > 0 && (
          <List sx={{ py: 0 }}>
            {manuals.map((manual, index) => (
              <ListItem
                key={manual.name || index}
                sx={{
                  borderRadius: 1.5,
                  mb: 0.5,
                  transition: 'all 0.2s',
                  '&:hover': {
                    bgcolor: alpha(theme.palette.primary.main, 0.08),
                    transform: 'translateX(4px)',
                  },
                }}
              >
                <ListItemText
                  primary={
                    <Typography
                      variant="body2"
                      sx={{
                        fontWeight: 500,
                        color: 'text.primary',
                        fontSize: '0.875rem',
                        lineHeight: 1.5,
                      }}
                    >
                      {manual.display_name || manual.name}
                    </Typography>
                  }
                  secondary={
                    <Typography
                      variant="caption"
                      sx={{
                        color: 'text.secondary',
                        fontSize: '0.75rem',
                        mt: 0.25,
                        display: 'block',
                      }}
                    >
                      {manual.name}
                    </Typography>
                  }
                />
              </ListItem>
            ))}
          </List>
        )}
          </Box>
        </Collapse>
        </Box>
      )}
    </Paper>
  );
};

export default ManualsList;

