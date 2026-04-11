// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CalendarView - Integrated Calendar Component
 *
 * Features:
 * - Day, week, and month views
 * - Email-linked events
 * - Drag-and-drop scheduling
 * - Event quick-create
 * - Availability visualization
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Plus,
  Calendar as CalendarIcon,
  Clock,
  MapPin,
  Users,
  Video,
  Mail,
  MoreHorizontal,
  X,
  Check,
  Edit,
  Trash2,
  Link,
  Bell,
  Repeat,
} from 'lucide-react';
import {
  useCalendar,
  type CalendarEvent,
  type Calendar,
  type TimeSlot,
} from '../../lib/calendar-focus';

// ============================================================================
// Types
// ============================================================================

type ViewType = 'day' | 'week' | 'month';

interface CalendarViewProps {
  onEventClick?: (event: CalendarEvent) => void;
  onSlotClick?: (start: Date, end: Date) => void;
  onCreateFromEmail?: (emailId: string) => void;
}

// ============================================================================
// Utilities
// ============================================================================

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatDate(date: Date): string {
  return date.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
}

function isSameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() &&
         a.getMonth() === b.getMonth() &&
         a.getDate() === b.getDate();
}

function getWeekDays(date: Date): Date[] {
  const start = new Date(date);
  start.setDate(start.getDate() - start.getDay());

  return Array.from({ length: 7 }, (_, i) => {
    const day = new Date(start);
    day.setDate(start.getDate() + i);
    return day;
  });
}

function getMonthDays(date: Date): Date[] {
  const start = new Date(date.getFullYear(), date.getMonth(), 1);
  const end = new Date(date.getFullYear(), date.getMonth() + 1, 0);

  // Start from Sunday of the first week
  start.setDate(start.getDate() - start.getDay());

  const days: Date[] = [];
  const current = new Date(start);

  while (current <= end || days.length % 7 !== 0) {
    days.push(new Date(current));
    current.setDate(current.getDate() + 1);
  }

  return days;
}

function getEventColor(event: CalendarEvent): string {
  if (event.color) return event.color;

  const colors = [
    'bg-blue-500',
    'bg-green-500',
    'bg-purple-500',
    'bg-orange-500',
    'bg-pink-500',
    'bg-teal-500',
  ];

  // Deterministic color based on event id
  const hash = event.id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return colors[hash % colors.length];
}

// ============================================================================
// Components
// ============================================================================

// Time Grid for Day/Week view
function TimeGrid({
  events,
  days,
  onEventClick,
  onSlotClick,
}: {
  events: CalendarEvent[];
  days: Date[];
  onEventClick?: (event: CalendarEvent) => void;
  onSlotClick?: (start: Date, end: Date) => void;
}) {
  const hours = Array.from({ length: 24 }, (_, i) => i);

  const getEventsForDayHour = (day: Date, hour: number) => {
    return events.filter(event => {
      if (!isSameDay(event.start, day)) return false;
      return event.start.getHours() <= hour && event.end.getHours() > hour;
    });
  };

  const handleSlotClick = (day: Date, hour: number) => {
    const start = new Date(day);
    start.setHours(hour, 0, 0, 0);
    const end = new Date(start);
    end.setHours(hour + 1);
    onSlotClick?.(start, end);
  };

  return (
    <div className="flex-1 overflow-auto">
      <div className="min-w-[800px]">
        {/* Header with day names */}
        <div className="flex border-b sticky top-0 bg-white dark:bg-gray-800 z-10">
          <div className="w-16 flex-shrink-0" />
          {days.map((day, idx) => (
            <div
              key={idx}
              className={`flex-1 p-2 text-center border-l ${
                isSameDay(day, new Date()) ? 'bg-blue-50 dark:bg-blue-900/20' : ''
              }`}
            >
              <p className="text-xs text-gray-500">
                {day.toLocaleDateString([], { weekday: 'short' })}
              </p>
              <p className={`text-lg font-semibold ${
                isSameDay(day, new Date()) ? 'text-blue-600' : ''
              }`}>
                {day.getDate()}
              </p>
            </div>
          ))}
        </div>

        {/* Time slots */}
        <div className="relative">
          {hours.map(hour => (
            <div key={hour} className="flex border-b h-16">
              <div className="w-16 flex-shrink-0 pr-2 text-right text-xs text-gray-500 -mt-2">
                {hour === 0 ? '' : `${hour.toString().padStart(2, '0')}:00`}
              </div>
              {days.map((day, dayIdx) => {
                const slotEvents = getEventsForDayHour(day, hour);
                return (
                  <div
                    key={dayIdx}
                    onClick={() => handleSlotClick(day, hour)}
                    className="flex-1 border-l relative cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50"
                  >
                    {slotEvents.map((event, eventIdx) => {
                      const startHour = event.start.getHours();
                      const isStart = startHour === hour;

                      if (!isStart) return null;

                      const duration = (event.end.getTime() - event.start.getTime()) / (1000 * 60 * 60);

                      return (
                        <div
                          key={event.id}
                          onClick={(e) => {
                            e.stopPropagation();
                            onEventClick?.(event);
                          }}
                          className={`absolute left-1 right-1 rounded px-2 py-1 text-white text-xs cursor-pointer hover:opacity-90 ${getEventColor(event)}`}
                          style={{
                            top: `${(event.start.getMinutes() / 60) * 100}%`,
                            height: `${duration * 64}px`,
                            zIndex: eventIdx + 1,
                          }}
                        >
                          <p className="font-medium truncate">{event.title}</p>
                          <p className="opacity-75 truncate">
                            {formatTime(event.start)} - {formatTime(event.end)}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Month Grid
function MonthGrid({
  events,
  days,
  currentMonth,
  onEventClick,
  onSlotClick,
}: {
  events: CalendarEvent[];
  days: Date[];
  currentMonth: number;
  onEventClick?: (event: CalendarEvent) => void;
  onSlotClick?: (start: Date, end: Date) => void;
}) {
  const getEventsForDay = (day: Date) => {
    return events.filter(event => isSameDay(event.start, day));
  };

  const handleDayClick = (day: Date) => {
    const start = new Date(day);
    start.setHours(9, 0, 0, 0);
    const end = new Date(day);
    end.setHours(10, 0, 0, 0);
    onSlotClick?.(start, end);
  };

  return (
    <div className="flex-1 overflow-auto p-4">
      {/* Day headers */}
      <div className="grid grid-cols-7 mb-2">
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
          <div key={day} className="text-center text-xs font-medium text-gray-500 py-2">
            {day}
          </div>
        ))}
      </div>

      {/* Day cells */}
      <div className="grid grid-cols-7 gap-1">
        {days.map((day, idx) => {
          const dayEvents = getEventsForDay(day);
          const isCurrentMonth = day.getMonth() === currentMonth;
          const isToday = isSameDay(day, new Date());

          return (
            <div
              key={idx}
              onClick={() => handleDayClick(day)}
              className={`min-h-[100px] p-1 border rounded cursor-pointer transition-colors ${
                isCurrentMonth
                  ? 'bg-white dark:bg-gray-800'
                  : 'bg-gray-50 dark:bg-gray-900 text-gray-400'
              } ${isToday ? 'ring-2 ring-blue-500' : ''} hover:bg-gray-50 dark:hover:bg-gray-700`}
            >
              <p className={`text-sm font-medium mb-1 ${
                isToday ? 'text-blue-600' : ''
              }`}>
                {day.getDate()}
              </p>
              <div className="space-y-1">
                {dayEvents.slice(0, 3).map(event => (
                  <div
                    key={event.id}
                    onClick={(e) => {
                      e.stopPropagation();
                      onEventClick?.(event);
                    }}
                    className={`text-xs truncate rounded px-1 py-0.5 text-white ${getEventColor(event)}`}
                  >
                    {event.title}
                  </div>
                ))}
                {dayEvents.length > 3 && (
                  <p className="text-xs text-gray-500">+{dayEvents.length - 3} more</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Event Detail Modal
function EventDetailModal({
  event,
  onClose,
  onEdit,
  onDelete,
}: {
  event: CalendarEvent;
  onClose: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-md m-4">
        <div className={`p-4 rounded-t-xl text-white ${getEventColor(event)}`}>
          <div className="flex items-start justify-between">
            <h2 className="text-xl font-semibold">{event.title}</h2>
            <button onClick={onClose} className="p-1 hover:bg-white/20 rounded">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="p-6 space-y-4">
          <div className="flex items-center gap-3">
            <Clock className="w-5 h-5 text-gray-400" />
            <div>
              <p className="font-medium">{formatDate(event.start)}</p>
              <p className="text-sm text-gray-500">
                {formatTime(event.start)} - {formatTime(event.end)}
              </p>
            </div>
          </div>

          {event.location && (
            <div className="flex items-center gap-3">
              <MapPin className="w-5 h-5 text-gray-400" />
              <p>{event.location}</p>
            </div>
          )}

          {event.meetingLink && (
            <div className="flex items-center gap-3">
              <Video className="w-5 h-5 text-gray-400" />
              <a
                href={event.meetingLink}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:underline"
              >
                Join Meeting
              </a>
            </div>
          )}

          {event.attendees.length > 0 && (
            <div className="flex items-start gap-3">
              <Users className="w-5 h-5 text-gray-400 mt-0.5" />
              <div className="space-y-1">
                {event.attendees.map((attendee, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <span>{attendee.name || attendee.email}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      attendee.status === 'accepted' ? 'bg-green-100 text-green-700' :
                      attendee.status === 'declined' ? 'bg-red-100 text-red-700' :
                      attendee.status === 'tentative' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {attendee.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {event.description && (
            <div className="pt-4 border-t">
              <p className="text-sm text-gray-600 dark:text-gray-400">{event.description}</p>
            </div>
          )}

          {event.sourceEmailId && (
            <div className="flex items-center gap-2 pt-4 border-t text-sm text-gray-500">
              <Mail className="w-4 h-4" />
              <span>Created from email</span>
            </div>
          )}
        </div>

        <div className="p-4 border-t flex justify-between">
          <button
            onClick={onDelete}
            className="flex items-center gap-2 px-4 py-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg"
          >
            <Trash2 className="w-4 h-4" />
            Delete
          </button>
          <button
            onClick={onEdit}
            className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            <Edit className="w-4 h-4" />
            Edit
          </button>
        </div>
      </div>
    </div>
  );
}

// Quick Create Event Modal
function QuickCreateModal({
  initialStart,
  initialEnd,
  onClose,
  onCreate,
}: {
  initialStart: Date;
  initialEnd: Date;
  onClose: () => void;
  onCreate: (event: Partial<CalendarEvent>) => void;
}) {
  const [title, setTitle] = useState('');
  const [start, setStart] = useState(initialStart);
  const [end, setEnd] = useState(initialEnd);
  const [location, setLocation] = useState('');
  const [allDay, setAllDay] = useState(false);

  const handleCreate = () => {
    if (!title.trim()) return;

    onCreate({
      title,
      start,
      end,
      location: location || undefined,
      allDay,
      attendees: [],
      status: 'confirmed',
      visibility: 'private',
      reminders: [{ type: 'notification', minutes: 15 }],
      calendar: 'default',
    });
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-md m-4">
        <div className="p-6 border-b">
          <h2 className="text-xl font-semibold">New Event</h2>
        </div>

        <div className="p-6 space-y-4">
          <div>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Event title"
              autoFocus
              className="w-full px-3 py-2 border rounded-lg text-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={allDay}
                onChange={(e) => setAllDay(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">All day</span>
            </label>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Start</label>
              <input
                type={allDay ? 'date' : 'datetime-local'}
                value={allDay
                  ? start.toISOString().split('T')[0]
                  : start.toISOString().slice(0, 16)
                }
                onChange={(e) => setStart(new Date(e.target.value))}
                className="w-full px-3 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">End</label>
              <input
                type={allDay ? 'date' : 'datetime-local'}
                value={allDay
                  ? end.toISOString().split('T')[0]
                  : end.toISOString().slice(0, 16)
                }
                onChange={(e) => setEnd(new Date(e.target.value))}
                className="w-full px-3 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Location</label>
            <input
              type="text"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              placeholder="Add location"
              className="w-full px-3 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
            />
          </div>
        </div>

        <div className="p-4 border-t flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!title.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}

// Mini Calendar for navigation
function MiniCalendar({
  currentDate,
  onDateSelect,
}: {
  currentDate: Date;
  onDateSelect: (date: Date) => void;
}) {
  const [viewDate, setViewDate] = useState(currentDate);
  const days = getMonthDays(viewDate);

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <button
          onClick={() => {
            const prev = new Date(viewDate);
            prev.setMonth(prev.getMonth() - 1);
            setViewDate(prev);
          }}
          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
        <span className="text-sm font-medium">
          {viewDate.toLocaleDateString([], { month: 'long', year: 'numeric' })}
        </span>
        <button
          onClick={() => {
            const next = new Date(viewDate);
            next.setMonth(next.getMonth() + 1);
            setViewDate(next);
          }}
          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      <div className="grid grid-cols-7 gap-1 text-center">
        {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((d, i) => (
          <div key={i} className="text-xs text-gray-500 py-1">{d}</div>
        ))}
        {days.map((day, idx) => (
          <button
            key={idx}
            onClick={() => onDateSelect(day)}
            className={`text-xs p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
              day.getMonth() !== viewDate.getMonth() ? 'text-gray-400' : ''
            } ${isSameDay(day, currentDate) ? 'bg-blue-500 text-white hover:bg-blue-600' : ''} ${
              isSameDay(day, new Date()) && !isSameDay(day, currentDate) ? 'text-blue-600 font-bold' : ''
            }`}
          >
            {day.getDate()}
          </button>
        ))}
      </div>
    </div>
  );
}

// Main Calendar View
export function CalendarView({
  onEventClick,
  onSlotClick,
  onCreateFromEmail,
}: CalendarViewProps) {
  const {
    events,
    currentDate,
    view,
    setCurrentDate,
    setView,
    createEvent,
    updateEvent,
    deleteEvent,
    calendars,
  } = useCalendar();

  const [selectedEvent, setSelectedEvent] = useState<CalendarEvent | null>(null);
  const [quickCreateSlot, setQuickCreateSlot] = useState<{ start: Date; end: Date } | null>(null);
  const [visibleCalendars, setVisibleCalendars] = useState<string[]>(
    calendars.filter(c => c.isVisible).map(c => c.id)
  );

  const viewDays = useMemo(() => {
    switch (view) {
      case 'day':
        return [currentDate];
      case 'week':
        return getWeekDays(currentDate);
      case 'month':
        return getMonthDays(currentDate);
    }
  }, [view, currentDate]);

  const visibleEvents = useMemo(() => {
    const start = viewDays[0];
    const end = viewDays[viewDays.length - 1];
    return events.filter(e =>
      visibleCalendars.includes(e.calendar) &&
      e.start <= end && e.end >= start
    );
  }, [events, viewDays, visibleCalendars]);

  const navigatePrev = () => {
    const newDate = new Date(currentDate);
    switch (view) {
      case 'day':
        newDate.setDate(newDate.getDate() - 1);
        break;
      case 'week':
        newDate.setDate(newDate.getDate() - 7);
        break;
      case 'month':
        newDate.setMonth(newDate.getMonth() - 1);
        break;
    }
    setCurrentDate(newDate);
  };

  const navigateNext = () => {
    const newDate = new Date(currentDate);
    switch (view) {
      case 'day':
        newDate.setDate(newDate.getDate() + 1);
        break;
      case 'week':
        newDate.setDate(newDate.getDate() + 7);
        break;
      case 'month':
        newDate.setMonth(newDate.getMonth() + 1);
        break;
    }
    setCurrentDate(newDate);
  };

  const handleSlotClick = (start: Date, end: Date) => {
    setQuickCreateSlot({ start, end });
    onSlotClick?.(start, end);
  };

  const handleEventClick = (event: CalendarEvent) => {
    setSelectedEvent(event);
    onEventClick?.(event);
  };

  const handleCreateEvent = (eventData: Partial<CalendarEvent>) => {
    createEvent(eventData as Omit<CalendarEvent, 'id' | 'createdAt' | 'updatedAt'>);
    setQuickCreateSlot(null);
  };

  const handleDeleteEvent = () => {
    if (selectedEvent) {
      deleteEvent(selectedEvent.id);
      setSelectedEvent(null);
    }
  };

  return (
    <div className="flex h-full bg-white dark:bg-gray-800">
      {/* Sidebar */}
      <div className="w-64 border-r flex flex-col">
        <div className="p-4">
          <button
            onClick={() => setQuickCreateSlot({
              start: new Date(),
              end: new Date(Date.now() + 60 * 60 * 1000)
            })}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 shadow-lg"
          >
            <Plus className="w-5 h-5" />
            Create Event
          </button>
        </div>

        <MiniCalendar
          currentDate={currentDate}
          onDateSelect={(date) => {
            setCurrentDate(date);
            setView('day');
          }}
        />

        <div className="flex-1 overflow-y-auto p-4 border-t">
          <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">Calendars</h3>
          {calendars.map(calendar => (
            <label
              key={calendar.id}
              className="flex items-center gap-2 p-2 hover:bg-gray-50 dark:hover:bg-gray-700 rounded cursor-pointer"
            >
              <input
                type="checkbox"
                checked={visibleCalendars.includes(calendar.id)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setVisibleCalendars(prev => [...prev, calendar.id]);
                  } else {
                    setVisibleCalendars(prev => prev.filter(id => id !== calendar.id));
                  }
                }}
                className="rounded"
                style={{ accentColor: calendar.color }}
              />
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: calendar.color }}
              />
              <span className="text-sm">{calendar.name}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setCurrentDate(new Date())}
              className="px-4 py-2 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
            >
              Today
            </button>
            <div className="flex items-center gap-1">
              <button
                onClick={navigatePrev}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              <button
                onClick={navigateNext}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
            <h2 className="text-xl font-semibold">
              {view === 'day' && formatDate(currentDate)}
              {view === 'week' && `${formatDate(viewDays[0])} - ${formatDate(viewDays[6])}`}
              {view === 'month' && currentDate.toLocaleDateString([], { month: 'long', year: 'numeric' })}
            </h2>
          </div>

          <div className="flex items-center gap-2">
            {(['day', 'week', 'month'] as ViewType[]).map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-4 py-2 rounded-lg capitalize ${
                  view === v
                    ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-100'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        {/* Calendar Grid */}
        {view === 'month' ? (
          <MonthGrid
            events={visibleEvents}
            days={viewDays}
            currentMonth={currentDate.getMonth()}
            onEventClick={handleEventClick}
            onSlotClick={handleSlotClick}
          />
        ) : (
          <TimeGrid
            events={visibleEvents}
            days={viewDays}
            onEventClick={handleEventClick}
            onSlotClick={handleSlotClick}
          />
        )}
      </div>

      {/* Event Detail Modal */}
      {selectedEvent && (
        <EventDetailModal
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
          onEdit={() => {
            // Would open edit modal
            setSelectedEvent(null);
          }}
          onDelete={handleDeleteEvent}
        />
      )}

      {/* Quick Create Modal */}
      {quickCreateSlot && (
        <QuickCreateModal
          initialStart={quickCreateSlot.start}
          initialEnd={quickCreateSlot.end}
          onClose={() => setQuickCreateSlot(null)}
          onCreate={handleCreateEvent}
        />
      )}
    </div>
  );
}

export default CalendarView;
