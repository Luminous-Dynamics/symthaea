// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Calendar View Component
 *
 * Email-integrated calendar with event extraction and scheduling
 */

import React, { useState, useEffect, useMemo } from 'react';

interface CalendarEvent {
  id: string;
  summary: string;
  description?: string;
  location?: string;
  organizer?: string;
  attendees: string[];
  startTime: string;
  endTime: string;
  allDay: boolean;
  status: 'tentative' | 'confirmed' | 'cancelled';
  responseStatus?: 'needs_action' | 'accepted' | 'declined' | 'tentative';
  source: 'email' | 'manual' | 'caldav' | 'google' | 'outlook';
  emailId?: string;
}

type ViewMode = 'month' | 'week' | 'day' | 'agenda';

export default function CalendarView() {
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentDate, setCurrentDate] = useState(new Date());
  const [viewMode, setViewMode] = useState<ViewMode>('month');
  const [selectedEvent, setSelectedEvent] = useState<CalendarEvent | null>(null);
  const [showEventModal, setShowEventModal] = useState(false);

  useEffect(() => {
    fetchEvents();
  }, [currentDate, viewMode]);

  const fetchEvents = async () => {
    setLoading(true);
    try {
      const start = getViewStart(currentDate, viewMode);
      const end = getViewEnd(currentDate, viewMode);
      const response = await fetch(
        `/api/calendar/events?start=${start.toISOString()}&end=${end.toISOString()}`
      );
      if (response.ok) setEvents(await response.json());
    } finally {
      setLoading(false);
    }
  };

  const getViewStart = (date: Date, mode: ViewMode): Date => {
    const d = new Date(date);
    if (mode === 'month') {
      d.setDate(1);
      d.setDate(d.getDate() - d.getDay());
    } else if (mode === 'week') {
      d.setDate(d.getDate() - d.getDay());
    }
    d.setHours(0, 0, 0, 0);
    return d;
  };

  const getViewEnd = (date: Date, mode: ViewMode): Date => {
    const d = new Date(date);
    if (mode === 'month') {
      d.setMonth(d.getMonth() + 1, 0);
      d.setDate(d.getDate() + (6 - d.getDay()));
    } else if (mode === 'week') {
      d.setDate(d.getDate() + (6 - d.getDay()));
    }
    d.setHours(23, 59, 59, 999);
    return d;
  };

  const navigatePrevious = () => {
    const d = new Date(currentDate);
    if (viewMode === 'month') d.setMonth(d.getMonth() - 1);
    else if (viewMode === 'week') d.setDate(d.getDate() - 7);
    else d.setDate(d.getDate() - 1);
    setCurrentDate(d);
  };

  const navigateNext = () => {
    const d = new Date(currentDate);
    if (viewMode === 'month') d.setMonth(d.getMonth() + 1);
    else if (viewMode === 'week') d.setDate(d.getDate() + 7);
    else d.setDate(d.getDate() + 1);
    setCurrentDate(d);
  };

  const goToToday = () => setCurrentDate(new Date());

  const handleEventClick = (event: CalendarEvent) => {
    setSelectedEvent(event);
    setShowEventModal(true);
  };

  const handleRespond = async (eventId: string, response: 'accepted' | 'declined' | 'tentative') => {
    await fetch(`/api/calendar/events/${eventId}/respond`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ response }),
    });
    fetchEvents();
    setShowEventModal(false);
  };

  const getTitle = (): string => {
    if (viewMode === 'month') {
      return currentDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
    } else if (viewMode === 'week') {
      const start = getViewStart(currentDate, 'week');
      const end = getViewEnd(currentDate, 'week');
      return `${start.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${end.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`;
    }
    return currentDate.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button onClick={goToToday} className="px-3 py-1.5 border border-border rounded hover:bg-muted/30">
            Today
          </button>
          <div className="flex items-center gap-1">
            <button onClick={navigatePrevious} className="p-2 hover:bg-muted/30 rounded">&lt;</button>
            <button onClick={navigateNext} className="p-2 hover:bg-muted/30 rounded">&gt;</button>
          </div>
          <h1 className="text-xl font-semibold">{getTitle()}</h1>
        </div>
        <div className="flex items-center gap-2">
          {(['day', 'week', 'month', 'agenda'] as ViewMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={`px-3 py-1.5 rounded capitalize ${
                viewMode === mode ? 'bg-primary text-white' : 'hover:bg-muted/30'
              }`}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      {/* Calendar Content */}
      <div className="flex-1 overflow-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
          </div>
        ) : viewMode === 'month' ? (
          <MonthView
            currentDate={currentDate}
            events={events}
            onEventClick={handleEventClick}
          />
        ) : viewMode === 'week' ? (
          <WeekView
            currentDate={currentDate}
            events={events}
            onEventClick={handleEventClick}
          />
        ) : viewMode === 'day' ? (
          <DayView
            currentDate={currentDate}
            events={events}
            onEventClick={handleEventClick}
          />
        ) : (
          <AgendaView events={events} onEventClick={handleEventClick} />
        )}
      </div>

      {/* Event Modal */}
      {showEventModal && selectedEvent && (
        <EventModal
          event={selectedEvent}
          onClose={() => setShowEventModal(false)}
          onRespond={handleRespond}
        />
      )}
    </div>
  );
}

function MonthView({
  currentDate,
  events,
  onEventClick,
}: {
  currentDate: Date;
  events: CalendarEvent[];
  onEventClick: (event: CalendarEvent) => void;
}) {
  const days = useMemo(() => {
    const result: Date[] = [];
    const start = new Date(currentDate);
    start.setDate(1);
    start.setDate(start.getDate() - start.getDay());

    for (let i = 0; i < 42; i++) {
      result.push(new Date(start));
      start.setDate(start.getDate() + 1);
    }
    return result;
  }, [currentDate]);

  const getEventsForDay = (date: Date) => {
    const dayStart = new Date(date);
    dayStart.setHours(0, 0, 0, 0);
    const dayEnd = new Date(date);
    dayEnd.setHours(23, 59, 59, 999);

    return events.filter((e) => {
      const start = new Date(e.startTime);
      const end = new Date(e.endTime);
      return start <= dayEnd && end >= dayStart;
    });
  };

  const isToday = (date: Date) => {
    const today = new Date();
    return date.toDateString() === today.toDateString();
  };

  const isCurrentMonth = (date: Date) => date.getMonth() === currentDate.getMonth();

  const weekDays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  return (
    <div className="h-full flex flex-col">
      <div className="grid grid-cols-7 border-b border-border">
        {weekDays.map((day) => (
          <div key={day} className="p-2 text-center text-sm font-medium text-muted">
            {day}
          </div>
        ))}
      </div>
      <div className="flex-1 grid grid-cols-7 grid-rows-6">
        {days.map((date, i) => {
          const dayEvents = getEventsForDay(date);
          return (
            <div
              key={i}
              className={`border-r border-b border-border p-1 min-h-24 ${
                !isCurrentMonth(date) ? 'bg-muted/20' : ''
              }`}
            >
              <div
                className={`text-sm mb-1 w-6 h-6 flex items-center justify-center rounded-full ${
                  isToday(date) ? 'bg-primary text-white' : ''
                }`}
              >
                {date.getDate()}
              </div>
              <div className="space-y-0.5">
                {dayEvents.slice(0, 3).map((event) => (
                  <div
                    key={event.id}
                    onClick={() => onEventClick(event)}
                    className={`text-xs px-1 py-0.5 rounded truncate cursor-pointer ${getEventColor(event)}`}
                  >
                    {event.allDay ? '' : formatTime(event.startTime) + ' '}
                    {event.summary}
                  </div>
                ))}
                {dayEvents.length > 3 && (
                  <div className="text-xs text-muted px-1">+{dayEvents.length - 3} more</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function WeekView({
  currentDate,
  events,
  onEventClick,
}: {
  currentDate: Date;
  events: CalendarEvent[];
  onEventClick: (event: CalendarEvent) => void;
}) {
  const days = useMemo(() => {
    const result: Date[] = [];
    const start = new Date(currentDate);
    start.setDate(start.getDate() - start.getDay());
    for (let i = 0; i < 7; i++) {
      result.push(new Date(start));
      start.setDate(start.getDate() + 1);
    }
    return result;
  }, [currentDate]);

  const hours = Array.from({ length: 24 }, (_, i) => i);

  const getEventsForDayHour = (date: Date, hour: number) => {
    return events.filter((e) => {
      if (e.allDay) return false;
      const start = new Date(e.startTime);
      return start.toDateString() === date.toDateString() && start.getHours() === hour;
    });
  };

  return (
    <div className="h-full flex flex-col overflow-auto">
      <div className="flex border-b border-border sticky top-0 bg-background z-10">
        <div className="w-16" />
        {days.map((date, i) => (
          <div key={i} className="flex-1 p-2 text-center border-l border-border">
            <div className="text-sm text-muted">{date.toLocaleDateString('en-US', { weekday: 'short' })}</div>
            <div className={`text-lg ${date.toDateString() === new Date().toDateString() ? 'text-primary font-bold' : ''}`}>
              {date.getDate()}
            </div>
          </div>
        ))}
      </div>
      <div className="flex-1">
        {hours.map((hour) => (
          <div key={hour} className="flex border-b border-border" style={{ minHeight: 48 }}>
            <div className="w-16 p-1 text-xs text-muted text-right pr-2">
              {hour === 0 ? '12 AM' : hour < 12 ? `${hour} AM` : hour === 12 ? '12 PM' : `${hour - 12} PM`}
            </div>
            {days.map((date, i) => {
              const hourEvents = getEventsForDayHour(date, hour);
              return (
                <div key={i} className="flex-1 border-l border-border p-0.5 relative">
                  {hourEvents.map((event) => (
                    <div
                      key={event.id}
                      onClick={() => onEventClick(event)}
                      className={`text-xs px-1 py-0.5 rounded cursor-pointer truncate ${getEventColor(event)}`}
                    >
                      {event.summary}
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function DayView({
  currentDate,
  events,
  onEventClick,
}: {
  currentDate: Date;
  events: CalendarEvent[];
  onEventClick: (event: CalendarEvent) => void;
}) {
  const dayEvents = events.filter((e) => {
    const start = new Date(e.startTime);
    return start.toDateString() === currentDate.toDateString();
  });

  const allDayEvents = dayEvents.filter((e) => e.allDay);
  const timedEvents = dayEvents.filter((e) => !e.allDay).sort((a, b) =>
    new Date(a.startTime).getTime() - new Date(b.startTime).getTime()
  );

  const hours = Array.from({ length: 24 }, (_, i) => i);

  return (
    <div className="h-full flex flex-col overflow-auto">
      {allDayEvents.length > 0 && (
        <div className="border-b border-border p-2">
          <div className="text-sm text-muted mb-2">All Day</div>
          {allDayEvents.map((event) => (
            <div
              key={event.id}
              onClick={() => onEventClick(event)}
              className={`px-2 py-1 rounded cursor-pointer mb-1 ${getEventColor(event)}`}
            >
              {event.summary}
            </div>
          ))}
        </div>
      )}
      <div className="flex-1">
        {hours.map((hour) => {
          const hourEvents = timedEvents.filter((e) => new Date(e.startTime).getHours() === hour);
          return (
            <div key={hour} className="flex border-b border-border" style={{ minHeight: 48 }}>
              <div className="w-20 p-2 text-sm text-muted">
                {hour === 0 ? '12 AM' : hour < 12 ? `${hour} AM` : hour === 12 ? '12 PM' : `${hour - 12} PM`}
              </div>
              <div className="flex-1 p-1">
                {hourEvents.map((event) => (
                  <div
                    key={event.id}
                    onClick={() => onEventClick(event)}
                    className={`px-2 py-1 rounded cursor-pointer mb-1 ${getEventColor(event)}`}
                  >
                    <div className="font-medium">{event.summary}</div>
                    <div className="text-sm opacity-80">
                      {formatTime(event.startTime)} - {formatTime(event.endTime)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AgendaView({
  events,
  onEventClick,
}: {
  events: CalendarEvent[];
  onEventClick: (event: CalendarEvent) => void;
}) {
  const sortedEvents = [...events]
    .filter((e) => new Date(e.startTime) >= new Date())
    .sort((a, b) => new Date(a.startTime).getTime() - new Date(b.startTime).getTime());

  const groupedByDate = sortedEvents.reduce((acc, event) => {
    const date = new Date(event.startTime).toDateString();
    if (!acc[date]) acc[date] = [];
    acc[date].push(event);
    return acc;
  }, {} as Record<string, CalendarEvent[]>);

  return (
    <div className="space-y-6">
      {Object.entries(groupedByDate).map(([date, dayEvents]) => (
        <div key={date}>
          <h3 className="font-semibold mb-2">
            {new Date(date).toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
          </h3>
          <div className="space-y-2">
            {dayEvents.map((event) => (
              <div
                key={event.id}
                onClick={() => onEventClick(event)}
                className="flex items-start gap-4 p-3 border border-border rounded-lg hover:bg-muted/30 cursor-pointer"
              >
                <div className="text-sm text-muted w-20">
                  {event.allDay ? 'All day' : `${formatTime(event.startTime)}`}
                </div>
                <div className="flex-1">
                  <div className="font-medium">{event.summary}</div>
                  {event.location && <div className="text-sm text-muted">{event.location}</div>}
                </div>
                {event.responseStatus && (
                  <span className={`px-2 py-0.5 rounded text-xs ${getResponseStatusColor(event.responseStatus)}`}>
                    {event.responseStatus.replace('_', ' ')}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
      {sortedEvents.length === 0 && (
        <div className="text-center py-12 text-muted">No upcoming events</div>
      )}
    </div>
  );
}

function EventModal({
  event,
  onClose,
  onRespond,
}: {
  event: CalendarEvent;
  onClose: () => void;
  onRespond: (eventId: string, response: 'accepted' | 'declined' | 'tentative') => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-lg">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <h2 className="text-lg font-semibold">{event.summary}</h2>
          <button onClick={onClose} className="text-muted hover:text-foreground">X</button>
        </div>
        <div className="p-4 space-y-4">
          <div className="flex items-center gap-3">
            <span className="text-muted w-20">When</span>
            <span>
              {event.allDay
                ? new Date(event.startTime).toLocaleDateString()
                : `${new Date(event.startTime).toLocaleDateString()} ${formatTime(event.startTime)} - ${formatTime(event.endTime)}`}
            </span>
          </div>
          {event.location && (
            <div className="flex items-center gap-3">
              <span className="text-muted w-20">Where</span>
              <span>{event.location}</span>
            </div>
          )}
          {event.organizer && (
            <div className="flex items-center gap-3">
              <span className="text-muted w-20">Organizer</span>
              <span>{event.organizer}</span>
            </div>
          )}
          {event.attendees.length > 0 && (
            <div className="flex items-start gap-3">
              <span className="text-muted w-20">Attendees</span>
              <div className="flex-1">
                {event.attendees.map((a, i) => (
                  <div key={i} className="text-sm">{a}</div>
                ))}
              </div>
            </div>
          )}
          {event.description && (
            <div>
              <span className="text-muted">Description</span>
              <p className="mt-1 text-sm whitespace-pre-wrap">{event.description}</p>
            </div>
          )}
          <div className="flex items-center gap-3">
            <span className="text-muted w-20">Status</span>
            <span className={`px-2 py-0.5 rounded text-sm ${getEventColor(event)}`}>{event.status}</span>
          </div>

          {event.responseStatus === 'needs_action' && (
            <div className="border-t border-border pt-4">
              <p className="text-sm text-muted mb-3">Will you attend?</p>
              <div className="flex gap-2">
                <button
                  onClick={() => onRespond(event.id, 'accepted')}
                  className="flex-1 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  Accept
                </button>
                <button
                  onClick={() => onRespond(event.id, 'tentative')}
                  className="flex-1 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600"
                >
                  Maybe
                </button>
                <button
                  onClick={() => onRespond(event.id, 'declined')}
                  className="flex-1 py-2 bg-red-500 text-white rounded hover:bg-red-600"
                >
                  Decline
                </button>
              </div>
            </div>
          )}

          {event.emailId && (
            <div className="border-t border-border pt-4">
              <a href={`/email/${event.emailId}`} className="text-primary hover:underline text-sm">
                View original email
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function getEventColor(event: CalendarEvent): string {
  if (event.status === 'cancelled') return 'bg-gray-200 text-gray-600 line-through';
  if (event.status === 'tentative') return 'bg-yellow-100 text-yellow-800';
  switch (event.source) {
    case 'google': return 'bg-blue-100 text-blue-800';
    case 'outlook': return 'bg-indigo-100 text-indigo-800';
    default: return 'bg-primary/10 text-primary';
  }
}

function getResponseStatusColor(status: string): string {
  switch (status) {
    case 'accepted': return 'bg-green-100 text-green-800';
    case 'declined': return 'bg-red-100 text-red-800';
    case 'tentative': return 'bg-yellow-100 text-yellow-800';
    default: return 'bg-gray-100 text-gray-800';
  }
}

function formatTime(isoString: string): string {
  return new Date(isoString).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}
